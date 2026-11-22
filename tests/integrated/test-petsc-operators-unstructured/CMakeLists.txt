bout_add_integrated_test(
  test_petsc_operators_unstructured
  SOURCES test_petsc_operators_unstructured.cxx
  USE_RUNTEST USE_DATA_BOUT_INP
  REQUIRES BOUT_HAS_PETSC
  REQUIRES BOUT_USE_METRIC_3D
  REQUIRES zoidberg_unstructured_FOUND
  PROCESSORS 2
)
