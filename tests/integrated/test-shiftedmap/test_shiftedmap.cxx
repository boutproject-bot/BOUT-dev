// Parallel slices and Grad_par of a field read from the grid file, with the
// parallel transform chosen in the input (shifted or shiftedmap). Writes,
// for each cell, the values of its forward and backward slices (which are
// stored at y + 1 and y - 1) so that runs on different layouts of the same
// cells can be compared cell by cell. Where field lines end on targets, a
// parallel Dirichlet condition sets the slices; the number of target points
// in each kind of parallel boundary region, and the number the default y
// boundary iterator visits, are written too.

#include "bout/bout.hxx"
#include "bout/boutcomm.hxx"
#include "bout/coordinates.hxx"
#include "bout/difops.hxx"
#include "bout/field3d.hxx"
#include "bout/globals.hxx"
#include "bout/mesh.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"
#include "bout/yboundary_regions.hxx"

#include <fmt/format.h>
#include <iterator>
#include <mpi.h>

namespace {
int globalSum(int local) {
  int total = 0;
  MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, BoutComm::get());
  return total;
}
} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);
  {
    Mesh* mesh = bout::globals::mesh;
    Field3D f{mesh};
    if (mesh->get(f, "f_test", 0.0, false) != 0) {
      throw BoutException("f_test not in the grid file");
    }
    mesh->communicate(f); // Also calculates the parallel slices
    // Value 0.3 at the targets (no effect where there are none)
    f.applyParallelBoundary("parallel_dirichlet_o3(0.3)");

    Field3D up{zeroFrom(f)};
    Field3D down{zeroFrom(f)};
    BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
      up[i] = f.yup()[i.yp()];
      down[i] = f.ydown()[i.ym()];
    }

    Options dump;
    dump["f_up"] = up;
    dump["f_down"] = down;
    dump["grad_par"] = Grad_par(f);

    for (const auto type : {BoundaryParType::xin_fwd, BoundaryParType::xin_bwd,
                            BoundaryParType::xout_fwd, BoundaryParType::xout_bwd}) {
      int points = 0;
      for (const auto& region : mesh->getBoundariesPar(type)) {
        points += static_cast<int>(std::distance(region->begin(), region->end()));
      }
      dump[fmt::format("points_{}", toString(type))] = globalSum(points);
    }
    int ybndry_up = 0;
    int ybndry_down = 0;
    f.getCoordinates()->getYBoundary().iter(
        [&](const auto& point) { (point.dir() > 0 ? ybndry_up : ybndry_down) += 1; });
    dump["ybndry_up"] = globalSum(ybndry_up);
    dump["ybndry_down"] = globalSum(ybndry_down);

    bout::writeDefaultOutputFile(dump);
  }
  BoutFinalise();
  return 0;
}
