/*
 * Shifted metric with the neighbour along the magnetic field from a map
 * (type = shiftedmap), for meshes whose cells are not stored in field-line
 * order along y. See ShiftedMetricMap in paralleltransform.hxx.
 */

#include "fci_comm.hxx"

#include "bout/paralleltransform.hxx"
#include <bout/boutexception.hxx>
#include <bout/constants.hxx>
#include <bout/fft.hxx>
#include <bout/mesh.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

ShiftedMetricMap::ShiftedMetricMap(Mesh& m, BoutReal zlength_in, Options* opt)
    : ParallelTransform(m, opt), zlength(zlength_in),
      gather(std::make_unique<GlobalField3DAccess>(&m)) {
  ShiftedMetricMap::checkInputGrid();

  if (mesh.ystart != 1) {
    throw BoutException("paralleltransform type shiftedmap needs MYG = 1, not {}",
                        mesh.ystart);
  }

  const int nmodes = mesh.LocalNz / 2 + 1;
  const std::array<std::pair<int, std::string>, 2> directions{
      {{+1, "forward"}, {-1, "backward"}}};
  for (const auto& [y_offset, name] : directions) {
    Field2D neighbour_x{&mesh};
    Field2D neighbour_y{&mesh};
    Field2D dzeta{&mesh};
    const std::string prefix = fmt::format("pmap_{}_", name);
    if (mesh.get(neighbour_x, prefix + "x", -1.0, false) != 0
        or mesh.get(neighbour_y, prefix + "y", -1.0, false) != 0
        or mesh.get(dzeta, prefix + "dzeta", 0.0, false) != 0) {
      throw BoutException("paralleltransform type shiftedmap needs {0}x, {0}y and "
                          "{0}dzeta in the grid file",
                          prefix);
    }

    Slice slice{y_offset, Matrix<int>(mesh.LocalNx, mesh.LocalNy),
                Tensor<dcomplex>(mesh.LocalNx, mesh.LocalNy, nmodes)};
    BOUT_FOR_SERIAL(i, mesh.getRegion2D("RGN_NOY")) {
      const int x = i.x();
      const int y = i.y();
      const auto gx = static_cast<int>(std::lround(neighbour_x[i]));
      if (gx < 0) {
        if (x >= mesh.xstart and x <= mesh.xend) {
          throw BoutException("shiftedmap: cell ({:d}, {:d}) has no {} neighbour; "
                              "parallel boundaries are not supported",
                              x, y, name);
        }
        slice.source(x, y) = -1;
        continue;
      }
      // Global y index with guard cells, as GlobalField3DAccess expects
      const auto gy = static_cast<int>(std::lround(neighbour_y[i])) + mesh.ystart;
      for (int z = 0; z < mesh.LocalNz; ++z) {
        gather->request(gather->xyzglobal(gx, gy, z));
      }
      slice.source(x, y) = gather->xyzglobal(gx, gy, 0).ind;

      // The slice is the neighbour's column at z + dzeta: each Fourier mode
      // gets the phase exp(i k dzeta)
      for (int jz = 0; jz < nmodes; ++jz) {
        const BoutReal kwave = jz * 2.0 * PI / zlength; // wave number is 1/[rad]
        slice.phase(x, y, jz) = dcomplex(cos(kwave * dzeta[i]), sin(kwave * dzeta[i]));
      }
    }
    slices.push_back(std::move(slice));
  }
}

ShiftedMetricMap::~ShiftedMetricMap() = default;

void ShiftedMetricMap::checkInputGrid() {
  std::string parallel_transform;
  if (mesh.isDataSourceGridFile()
      and mesh.get(parallel_transform, "parallel_transform") == 0
      and parallel_transform != "shiftedmap") {
    throw BoutException("Incorrect parallel transform type '{}' used to generate the "
                        "grid for ShiftedMetricMap. Should be 'shiftedmap'.",
                        parallel_transform);
  }
}

void ShiftedMetricMap::calcParallelSlices(Field3D& f) {
  ASSERT1(f.getDirectionY() == YDirectionType::Standard);
  f.splitParallelSlices();

  const auto columns = gather->communicate(f);
  const int nz = mesh.LocalNz;
  const int nmodes = nz / 2 + 1;
  Array<dcomplex> modes(nmodes);
  std::vector<BoutReal> column(nz);

  for (const auto& slice : slices) {
    Field3D& f_slice = f.ynext(slice.y_offset);
    f_slice.allocate();

    // Values for cell (x, y) are stored at (x, y + y_offset), as in the
    // other transforms
    BOUT_FOR_SERIAL(i, mesh.getRegion2D("RGN_NOY")) {
      const int x = i.x();
      const int y = i.y();
      BoutReal* out = &f_slice(x, y + slice.y_offset, 0);
      const int source = slice.source(x, y);
      if (source < 0) {
        // x guard cell without a neighbour: copy the cell's own column
        std::copy_n(&f(x, y, 0), nz, out);
        continue;
      }
      for (int z = 0; z < nz; ++z) {
        column[z] = columns[IndG3D(source + z)];
      }
      rfft(column.data(), nz, &modes[0]);
      for (int jz = 1; jz < nmodes; ++jz) {
        modes[jz] *= slice.phase(x, y, jz);
      }
      irfft(&modes[0], nz, out);
    }
  }
}
