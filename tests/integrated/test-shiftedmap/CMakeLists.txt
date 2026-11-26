bout_add_integrated_test(
  test-shiftedmap
  SOURCES test_shiftedmap.cxx
  USE_RUNTEST USE_DATA_BOUT_INP
  REQUIRES BOUT_USE_METRIC_3D
  PROCESSORS 2
)
