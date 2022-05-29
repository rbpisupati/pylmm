## Wrapper function written by Rahul Pisupati for performing GWAS
## input: 
#   SNPs from HDF5 file (SNPmatch, snp_genotype class)
#   Phenotypes are in a pandas dataframe

import numpy as np
import scipy as sp
import pandas as pd
import h5py as h5
from snpmatch.core import snp_genotype
from . import kinship

import logging
log = logging.getLogger(__name__)

def getMaf(snp):
    ## Give a 2d array with snps in rows and accs in columns
    alt_freq = np.mean(snp, axis=1)
    ref_freq = 1 - alt_freq
    return(np.minimum.reduce([alt_freq, ref_freq]))

def get_bh_thres(pvals, fdr_thres=0.05):
    """
    Implements Benjamini-Hochberg FDR threshold (1995)
    """
    m = len(pvals)   
    s_pvals = sorted(pvals) 
    for i, p in enumerate(s_pvals):
        thes_pval = ((i + 1.0) / float(m)) * fdr_thres
        if p > thes_pval:
            break
        
    return thes_pval

def get_bhy_thres(pvals, fdr_thres=0.05):
    """
    Implements the Benjamini-Hochberg-Yekutieli procedure (2001).
    Assumes arbitrary dependence between variables.
    """
    #Euler-Mascheroni constant 
    gamma = 0.57721566490153286060651209008    
    m = len(pvals)   
    m_float = float(m) 
    s_pvals = sorted(pvals) 
    s = 1.0
    for i, p in enumerate(s_pvals):
        if i > 2:
            s = s + 1.0/(i-1)
        thes_pval = ((i + 1.0) / m_float) * fdr_thres / s
        if p > thes_pval:
            break        
    return thes_pval


class GWAS(object):
    """
    Class object for perform GWAS using limix (scan function)

    Parses SNP array in hdf5 using SNPmatch snp_genotype
    """
    def __init__(self, genotype_file, phenos_df = None, chunk_size = 50000):
        self.g = self.load_genotype_file( genotype_file )
        if phenos_df is not None:
            self.load_phenotype( phenos_df )
        self.chunk_size = chunk_size

    def load_genotype_file(self, genotype_file):
        log.info("loading genotype file")
        g = snp_genotype.load_genotype_files(genotype_file)
        return(g)

    def load_phenotype(self, phenos_df):
        assert type(phenos_df) is pd.DataFrame, "provide a dataframe 1st column as accession ID"
        filtered_pheno = phenos_df.copy()
        filtered_pheno.iloc[:,0] = filtered_pheno.iloc[:,0].astype(str) 
        filtered_pheno = filtered_pheno.set_index( filtered_pheno.columns[0] )
        filtered_pheno = filtered_pheno.reindex( self.g.accessions )
        # with pd.option_context('mode.use_inf_as_null', True):
            # df = df.dropna()
        finite_ix = np.where(  (~np.isfinite( filtered_pheno )).sum(1) == 0 )[0]
        self.pheno = filtered_pheno.copy()
        self.finite_ix = finite_ix

    def load_kinship_file(self, kinship_file):
        log.info("loading kinship file")
        self.K = kinship.load_kinship_from_file( kinship_file, self.g.accessions )
        assert np.array_equal(self.g.accessions, np.array(self.K['accessions'])), "please provide a kinship matrix with same order as G"

    
    def filter_snps_by_maf(self, maf_threshold = 0.05, filter_pos_ix = None):
        """
        Iterator filtering SNPs by MAF
        """
        if filter_pos_ix is None:
            filter_pos_ix = np.arange(0, self.g.g.num_snps)
        assert type(filter_pos_ix) is np.ndarray, "provide an numpy array for required positions"
        num_snps_to_process = filter_pos_ix.shape[0]
        for ef_snp_start_ix in np.arange(0, num_snps_to_process, self.chunk_size):
            ef_snp_end_ix = min(num_snps_to_process, ef_snp_start_ix + self.chunk_size)
            ef_snp_ix = filter_pos_ix[np.arange(ef_snp_start_ix, ef_snp_end_ix)]
            ef_snp = self.g.g.snps[ef_snp_ix,:][:,self.finite_ix]
            ef_maf = getMaf(ef_snp)
            ef_maf_filter =  np.where(ef_maf >= maf_threshold)[0]
            yield( (ef_snp[ef_maf_filter,:], ef_snp_ix[ef_maf_filter]  ) )


    def perform_mtmm(self, maf_threshold = 0.05, filter_pos_ix = None, output_file = None, model_args = {}):
        """
        Perform a Multitrait mixed model using Limix

        """
        import limix
        K = self.K['k'][:,self.finite_ix][self.finite_ix,:]
        Y = self.pheno.iloc[self.finite_ix,:].values
        if Y.shape[1] != 2:
            NotImplementedError("currently model only accepts two traits")
        A = np.matrix('0 1; 1 0')
        A0 = np.ones((2, 1)) 
        A1 = np.eye(2)

        num_snps = self.g.g.num_snps
        betas = np.zeros(0, dtype = float)
        pvals = np.zeros((0, 3), dtype = float)
        mafs = np.zeros(0, dtype = int)
        maf_filter_ix = np.zeros(0, dtype = int)
        

        log.info("performing MTMM")
        tracker = 0
        for ef_snp in self.filter_snps_by_maf(maf_threshold = maf_threshold, filter_pos_ix = filter_pos_ix):
            tracker += 1
            ef_model = limix.qtl.scan(G = ef_snp[0].T, Y = Y, K = K, A = A, A0=A0, A1=A1, verbose=False, **model_args)
            pvals = np.append(pvals, ef_model.stats.loc[:,['pv10', 'pv20', 'pv21']].values, axis = 0)
            mafs = np.append(mafs, getMaf( ef_snp[0] ) )
            maf_filter_ix = np.append(maf_filter_ix, ef_snp[1] )

            if tracker % 10 == 0:
                log.info( "progress: %s SNPs processed" % maf_filter_ix.shape[0] )

        if output_file is not None:
            self.write_mapping_output(
                model="mtmm", 
                pvals= pvals, 
                betas=betas,
                maf = mafs,
                filtered_indices=maf_filter_ix, 
                output_file=output_file,
                pval_col_names = ['g', 'g_and_gxe', 'only_gxe']
            )
        return((pvals, maf_filter_ix))



    def perform_lmm(self, maf_threshold = 0.05, filter_pos_ix = None, output_file = None, model_args = {}):
        """
        perform association mapping using simple LMM
        """
        from . import lmm
        K = self.K['k'][:,self.finite_ix][self.finite_ix,:]
        Kva,Kve = sp.linalg.eigh( K )
        Y = self.pheno.iloc[:,0].values[self.finite_ix]
        TS = np.zeros(0, dtype = float)
        PS = np.zeros(0, dtype = float)
        MAF = np.zeros(0, dtype = float)
        MAF_ix = np.zeros(0, dtype = int)
        
        log.info("performing GWAS")
        tracker = 0
        for ef_snp in self.filter_snps_by_maf(maf_threshold, filter_pos_ix = filter_pos_ix): 
            ef_model = lmm.association_mapping(Y = Y, X = ef_snp[0].T, K = K, Kva=Kva, Kve=Kve, **model_args)
            TS = np.append(TS, ef_model[0])
            PS = np.append(PS, ef_model[1])
            MAF = np.append(MAF, getMaf(ef_snp[0]))
            MAF_ix = np.append(MAF_ix, ef_snp[1])
            if tracker % 10 == 0:
                log.info( "progress: %s SNPs processed" % MAF_ix.shape[0] )
        if output_file is not None:
            self.write_mapping_output(model="lmm", pvals= PS, mafs = MAF, betas=TS, filtered_indices=MAF_ix, output_file=output_file)
        return((TS, PS, MAF, MAF_ix))

    def write_mapping_output(self, model, pvals, mafs, betas, filtered_indices, output_file, pval_col_names = None):
        """
        Simple function to write output stats from association mapping

        """
        log.info("writing output file")
        output_h5 = h5.File(output_file, 'w')
        output_h5.create_dataset('model', compression="gzip", data= np.array([model], dtype = 'S') )
        output_h5.create_dataset('phenotype', compression="gzip", data=self.pheno.iloc[self.finite_ix,:].values)
        output_h5.create_dataset('num_snp', compression="gzip", data=np.array([self.g.g.num_snps]) )
        output_h5.create_dataset('accessions', compression="gzip", data= np.array(self.g.accessions[self.finite_ix],dtype='S') )
        output_h5.create_dataset('filtered_pos_ixs', compression="gzip", data = filtered_indices, chunks = True, dtype="int")
        output_h5.create_dataset('pvalues', compression="gzip", data = pvals, chunks = True, dtype="float", fillvalue = np.nan)
        output_h5.create_dataset('maf', compression="gzip", data = mafs, chunks = True, dtype="float", fillvalue = np.nan)
        if pval_col_names is not None:
            output_h5.create_dataset('pvalue_cols', compression="gzip", data = np.array(pval_col_names, dtype="S") )

        output_h5.create_dataset('betas', compression="gzip", data = betas, chunks = True, dtype="float", fillvalue = np.nan)
        
        output_h5.close()


    def load_gwas_result(self, result_file, output_file = None):
        """
        Function to load the results file of GWAS
        If you provide an output file, you could also call GWAS peaks
        """
        log.info("loading gwas results file")
        out_lmm = h5.File(result_file, 'r')
        filtered_indices = np.array(out_lmm['filtered_pos_ixs']).astype(int)
        out_data = {}
        out_data['num_snps'] = len(filtered_indices)
        out_data['data'] = pd.DataFrame( index = filtered_indices )
        out_data['data']['chr'] = np.array(self.g.g.chromosomes)[filtered_indices]
        out_data['data']['pos'] = self.g.g.positions[filtered_indices]
        # out_data['data']['beta'] = np.array(out_lmm['betas'])
        # out_data['data']['maf'] = np.array(out_lmm['maf'])
        with np.errstate(invalid='ignore'):
            pvals =  np.array(out_lmm['pvalues'])
            logpval = -np.log10(pvals)
        if 'pvalue_cols' in out_lmm.keys():
            pvalue_cols = out_lmm['pvalue_cols'][:].astype('U')
            for ef_model_ix in range( pvalue_cols.shape[0]) :
                out_data['data'].loc[:,'pval_' + pvalue_cols[ef_model_ix]] = pvals[:,ef_model_ix]
                out_data['data'].loc[:,'logpval_' + pvalue_cols[ef_model_ix]] = logpval[:,ef_model_ix]
        else:
            out_data['data']['pvalue'] = pvals
            out_data['data']['logpvalue'] = logpval
            out_data['bh_thres'] = get_bh_thres( pvals )
            if output_file is not None:
                # Call peaks and write peaks to the file
                out_data['data']['pval_filter'] = out_data['data']['logpvalue'] > -np.log10(out_data['bh_thres'])
                output_peaks = out_data['data'][out_data['data']['pval_filter']]
                if (output_peaks.shape[0] < out_data['num_snps'] / 100) and (output_peaks.shape[0] > 0):
                    ## do something about this,, more then 1% of the SNPs are highly significant.
                    output_peaks.loc[:,['chr', 'pos', 'logpvalue', 'beta']].to_csv(output_file, header = None, index = None)
        return(out_data)

