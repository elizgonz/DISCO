import os
import tempfile
import unittest

import h5py
import numpy as np

from export_params import MAX_DEG, N_MU_DENSE, export_params
from make_lines import LineProfileGenerator


LINES = ("Fe5250", "Fe6152", "Fe6173")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class ExportParamsTest(unittest.TestCase):
    def assert_export_matches_direct_profiles(self, path, line, generator):
        with h5py.File(path, "r") as params:
            mu_grid = params["mu_grid"][:]
            self.assertEqual(params.attrs["line"], line)
            self.assertEqual(int(params.attrs["n_mu"]), N_MU_DENSE)
            self.assertEqual(len(mu_grid), N_MU_DENSE)
            self.assertAlmostEqual(
                np.degrees(np.arccos(float(mu_grid[0]))), MAX_DEG, places=5)
            self.assertEqual(int(params.attrs["n_wave"]), len(generator.wl))
            self.assertEqual(
                params["wavelength"].shape, (len(generator.wl),))

            for component in ("GT", "OGR", "IgL"):
                group = params[f"pca/{component}"]
                self.assertEqual(
                    group["pca_coeff_grid"].shape,
                    (generator.eigenprofiles_gt.shape[0], N_MU_DENSE),
                )
                for dataset in group.values():
                    self.assertTrue(np.all(np.isfinite(dataset[:])))

            for name in ("epsilon1", "epsilon2"):
                group = params[f"distributions/{name}"]
                for field in ("alpha", "zeta", "omega", "median"):
                    self.assertEqual(group[field].shape, (N_MU_DENSE,))
                    self.assertTrue(np.all(np.isfinite(group[field][:])))

            for mu in np.linspace(mu_grid[0], 1.0, 21):
                mu = np.float32(mu)
                degree = min(
                    float(np.degrees(np.arccos(float(mu)))), MAX_DEG)
                python_profile, filling_factors = (
                    generator.get_full_profile(degree, 0.5, 0.5))

                pos = (float(mu) - float(mu_grid[0])) / (
                    float(mu_grid[-1]) - float(mu_grid[0])
                ) * (N_MU_DENSE - 1)
                index = min(int(np.floor(pos)), N_MU_DENSE - 2)
                fraction = np.float32(pos - index)

                components = {}
                for component in ("GT", "OGR", "IgL"):
                    group = params[f"pca/{component}"]
                    coeff_grid = group["pca_coeff_grid"][:]
                    coeff = coeff_grid[:, index] + fraction * (
                        coeff_grid[:, index + 1] - coeff_grid[:, index])
                    components[component] = (
                        group["mean_profile"][:]
                        + coeff @ group["eigenprofiles"][:]
                    )

                gt, igl, ogr = filling_factors
                exported_profile = (
                    components["GT"] * gt
                    + components["OGR"] * ogr
                    + components["IgL"] * igl
                )
                direct_normalized = python_profile / np.max(python_profile)
                exported_normalized = (
                    exported_profile / np.max(exported_profile))
                np.testing.assert_allclose(
                    exported_normalized,
                    direct_normalized,
                    rtol=0.0,
                    atol=1e-6,
                    err_msg=f"{line} profile mismatch at mu={mu}",
                )

    def test_exported_profiles_match_for_all_lines(self):
        for line in LINES:
            with self.subTest(line=line):
                generator = LineProfileGenerator(line)
                data_path = os.path.join(DATA_DIR, f"disco_{line}_params.h5")
                self.assertTrue(
                    os.path.isfile(data_path),
                    f"Expected exported model at {data_path}",
                )

                self.assert_export_matches_direct_profiles(
                    data_path, line, generator)

                with tempfile.NamedTemporaryFile(suffix=".h5") as output:
                    export_params(
                        output.name, generator=generator, line=line)
                    self.assert_export_matches_direct_profiles(
                        output.name, line, generator)


if __name__ == "__main__":
    unittest.main()
