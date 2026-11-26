bout_add_integrated_test(
  test-unstructured-planes
  SOURCES test_unstructured_planes.cxx
  USE_RUNTEST USE_DATA_BOUT_INP
  REQUIRES BOUT_HAS_PETSC
  REQUIRES BOUT_USE_METRIC_3D
  REQUIRES zoidberg_unstructured_FOUND
  PROCESSORS 2
)
