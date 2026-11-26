bout_add_integrated_test(
  test-jacobian-pattern-unstructured
  SOURCES test_jacobian_pattern_unstructured.cxx
  USE_RUNTEST USE_DATA_BOUT_INP
  REQUIRES BOUT_HAS_PETSC
  REQUIRES BOUT_USE_METRIC_3D
  REQUIRES zoidberg_unstructured_FOUND
  PROCESSORS 2
)

configure_file(../jacobian_tools/read_jacobian.py read_jacobian.py COPYONLY)
configure_file(../jacobian_tools/PetscBinaryIO.py PetscBinaryIO.py COPYONLY)
