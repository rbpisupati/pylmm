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
from . import lmm

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
    def __init__(self, genotype_file, phenos_df = None, chunk_size = 5000):
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
        finite_ix = np.where(np.isfinite( filtered_pheno.iloc[:,0] ))[0]
        self.pheno = filtered_pheno.copy()
        self.finite_ix = finite_ix

    def load_kinship_file(self, kinship_file):
        log.info("loading kinship file")
        self.K = kinship.load_kinship_from_file( kinship_file, self.g.accessions )

    def perform_gwas(self, output_file = None):
        """
        perform association mapping 
        """
        K = self.K['k'][:,self.finite_ix][self.finite_ix,:]
        Kva,Kve = sp.linalg.eigh( K )
        Y = self.pheno.iloc[:,0].values[self.finite_ix]
        TS = np.zeros(0, dtype = float)
        PS = np.zeros(0, dtype = float)
        MAF = np.zeros(0, dtype = float)
        
        log.info("performing GWAS")
        for ef_snp_ix in np.arange(0, self.g.g.num_snps, self.chunk_size):
            ef_snp_end_ix = min(self.g.g.num_snps, ef_snp_ix + self.chunk_size)
            ef_snp = self.g.g.snps[ef_snp_ix:ef_snp_end_ix,:][:,self.finite_ix]
            ef_model = lmm.association_mapping(Y = Y, X = ef_snp.T, K = K, Kva=Kva, Kve=Kve)
            TS = np.append(TS, ef_model[0])
            PS = np.append(PS, ef_model[1])
            MAF = np.append(MAF, getMaf(ef_snp))
            if ef_snp_ix % (self.chunk_size * 10) == 0:
                log.info( "progress: %s SNPs" % str(ef_snp_ix + self.chunk_size) )

        log.info("writing output file")
        if output_file is not None:
            output_h5 = h5.File(output_file, 'w')
            output_h5.create_dataset('phenotype', compression="gzip", data=Y)
            output_h5.create_dataset('num_snp', compression="gzip", data=np.array([self.g.g.num_snps]) )
            output_h5.create_dataset('accessions', compression="gzip", data= np.array(self.g.accessions[self.finite_ix],dtype='S') )
            output_h5.create_dataset('pvalues', compression="gzip", data = PS, chunks = True, dtype="float", fillvalue = np.nan)
            output_h5.create_dataset('betas', compression="gzip", data = TS, chunks = True, dtype="float", fillvalue = np.nan)
            output_h5.create_dataset('maf', compression="gzip", data = MAF, chunks = True, dtype="float", fillvalue = np.nan)
            output_h5.close()
        log.info("finished")
        return((TS, PS, MAF))

    def load_output_result(self, output_file, maf_filter = 0.05):
        out_lmm = h5.File(output_file, 'r')
        out_data = {}
        out_data['data'] = pd.DataFrame( {"chr": self.g.g.chromosomes, "pos": self.g.g.positions} )
        out_data['data']['pvalue'] = np.array(out_lmm['pvalues'])
        with np.errstate(invalid='ignore'):
            out_data['data']['logpvalue'] = -np.log10(out_data['data']['pvalue'])
        out_data['data']['beta'] = -np.log10(np.array(out_lmm['betas']))
        out_data['data']['maf'] = np.array(out_lmm['maf'])
        out_data['data']['maf_filter'] = out_data['data']['maf'] > maf_filter
        out_data['bh_thres'] = get_bh_thres( out_data['data']['pvalue'] )
        return(out_data)




#     #h5file.create_dataset('bhy_thres', compression="gzip", data=.transformation, shape=((1,)))
#     log.info("generating qqplot!")
#     plot.qqplot(np.array(lmm_pvals)[np.where(np.array(mafs) >= maf_thres)[0]], args['outFile'] + ".qqplot.png")
#     
#     
