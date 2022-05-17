# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2015  Nicholas A. Furlotte (nick.furlotte@gmail.com)

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

__version__ = '0.9.9'
__date__ = '05.05.2022'
__updated__ = '05.05.2022'


import os, sys
import pandas as pd
import numpy as np
import argparse
import logging
from pylmm.core import gwas


def setLog(logDebug):
    log = logging.getLogger()
    if logDebug:
        numeric_level = getattr(logging, "DEBUG", None)
    else:
        numeric_level = getattr(logging, "ERROR", None)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    lch = logging.StreamHandler()
    lch.setLevel(numeric_level)
    lch.setFormatter(log_format)
    log.setLevel(numeric_level)
    log.addHandler(lch)

def die(msg):
    sys.stderr.write('Error: ' + msg + '\n')
    sys.exit(1)



def get_options(program_license,program_version_message):
    inOptions = argparse.ArgumentParser(description=program_license)
    inOptions.add_argument('-V', '--version', action='version', version=program_version_message)

    subparsers = inOptions.add_subparsers(title='subcommands',description='Choose a command to run',help='Following commands are supported')
    
    gwas_parser = subparsers.add_parser('gwas', help="Perform a GWAS")
    gwas_parser.add_argument("-p", "--pheno_file", dest="phenotypes", help="phenotype file")
    gwas_parser.add_argument("-g", "--geno_hdf5", default = None, dest="genotypes", help="Path to SNP matrix given in binary hdf5 file")
    gwas_parser.add_argument("-k", "--kinship", default = None, dest="kinship", help="Path to kinship matrix for random")
    gwas_parser.add_argument("-v", "--verbose", action="store_true", dest="logDebug", default=False, help="Show verbose debugging output")
    gwas_parser.add_argument("-o", "--output", dest="outFile", default="identify_inbred", help="Output file from gwas")
    gwas_parser.set_defaults(func=perform_gwas)

    peaks_parser = subparsers.add_parser('get_peaks', help="Identify peaks for GWAS")
    peaks_parser.add_argument("-r", "--gwas_result", dest="gwas_result", help="LMM results file")
    peaks_parser.add_argument("-g", "--geno_hdf5", default = None, dest="genotypes", help="Path to SNP matrix given in binary hdf5 file")
    peaks_parser.add_argument("--maf_thres", default = 0.03, dest="maf_filter", help="Minor allel frequency threshold")
    peaks_parser.add_argument("-v", "--verbose", action="store_true", dest="logDebug", default=False, help="Show verbose debugging output")
    peaks_parser.add_argument("-o", "--output", dest="outFile", default="identify_inbred", help="Output file from gwas")
    peaks_parser.set_defaults(func=gwas_peaker)
    
    return inOptions



def perform_gwas(args):
  logging.info( "reading input phenotype file" )
  pheno_df = pd.read_csv( args['phenotypes'], header=None )
  pheno_df.iloc[:,0] = pheno_df.iloc[:,0].astype(int).astype(str)
  lmm = gwas.GWAS(args['genotypes'], phenos_df=pheno_df)
  lmm.load_kinship_file( args['kinship'] )
  if len(lmm.finite_ix) < 50:
    die( "provide phenotype file with information on more than 50 lines" )

  _ = lmm.perform_gwas( args['outFile'] )  


def gwas_peaker(args):
  snps_class = gwas.GWAS(args['genotypes'], phenos_df=None)
  snps_class.load_gwas_result( args['gwas_result'], maf_filter = args['maf_filter'], output_file = args['outFile'] )


def main():
  ''' Command line options '''
  program_version = "v%s" % __version__
  program_build_date = str(__updated__)
  program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
  program_shortdesc = "The main module for SNPmatch"
  program_license = '''%s
  Created by Rahul Pisupati on %s.
  Copyright 2016 Gregor Mendel Institute. All rights reserved.
  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.
USAGE
''' % (program_shortdesc, str(__date__))

  parser = get_options(program_license,program_version_message)
  args = vars(parser.parse_args())
  setLog(args['logDebug'])
  if 'func' not in args:
    parser.print_help()
    return(0)
  try:
    args['func'](args)
    return(0)
  except KeyboardInterrupt:
    return(0)
  except Exception as e:
    logging.exception(e)
    return(2)

if __name__=='__main__':
  sys.exit(main())