# src/app/config/paths.py
"""
Path configurations for data and files.
These are base paths used internally by settings.py for environment-based switching.
No side effects on import.
"""

import os

#--------------------------------------------------------
# Root definition
ROOT='../../..'

#--------------------------------------------------------
# Parent repository
# Assume that the parent repo is openapi_multiagents, and the folder workspace 
# contains the children repos. All repos are in openapi_multiagents/workspace/
PARENT_REPO = f"{ROOT}/../.."
DATA_DIRECTORY = f"{PARENT_REPO}/Dataset"

# Specific dataset paths
TSPEC_DATA_TEST_FILE = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean/Rel-18/28_series/28532-i00.md"
TSPEC_DATA_TEST = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean/Rel-18/28_series/"
TSPEC_DATA_PROD = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean"

#--------------------------------------------------------
# Main Paths repository
SRC_DIRECTORY=f"{ROOT}/src"
FILES_DIRECTORY=f"{ROOT}/files"

##--------------------------------------------------------
# src Paths
APP_DIRECTORY=f"{SRC_DIRECTORY}/app"

##--------------------------------------------------------
#files Paths
CHUNKS_DIRECTORY=f"{FILES_DIRECTORY}/chunks"

# Chunk file paths
CHUNKS_FILE_TEST_FILE = f"{CHUNKS_DIRECTORY}/tspec_chunks_test_rel_18_28_28532.pkl" # Use this for testing with Rel-18 data
CHUNKS_FILE_TEST = f"{CHUNKS_DIRECTORY}/tspec_chunks_test_rel_18_28.pkl" # Use this for testing with Rel-18 data
CHUNKS_FILE_PROD = f"{CHUNKS_DIRECTORY}/tspec_chunks_prod.pkl"

##--------------------------------------------------------
# Qdrant usage
COLLECTION_NAME_TEST_FILE = '3gpp_rel18_28_28532'
COLLECTION_NAME_TEST = '3gpp_rel18_28'
COLLECTION_NAME_PROD = '3gpp'