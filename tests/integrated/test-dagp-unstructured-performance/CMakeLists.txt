bout_add_integrated_test(
  test_dagp_unstructured_performance
  SOURCES test_dagp_unstructured_performance.cxx
  USE_RUNTEST USE_DATA_BOUT_INP
  REQUIRES BOUT_HAS_PETSC
  REQUIRES BOUT_USE_METRIC_3D
  REQUIRES zoidberg_unstructured_FOUND
)
