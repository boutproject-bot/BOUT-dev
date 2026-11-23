bout_add_mms_test(
  MMS-spatial-fci-unstructured
  SOURCES fci_mms.cxx
  USE_RUNTEST USE_DATA_BOUT_INP
  REQUIRES zoidberg_unstructured_FOUND
  REQUIRES BOUT_USE_METRIC_3D
  REQUIRES BOUT_HAS_PETSC
  PROCESSORS 2
  EXTRA_FILES data_cyl/BOUT.inp data_cyl_sheared/BOUT.inp
)
