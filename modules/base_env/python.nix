{ pkgs ? import <nixpkgs> { } }:
let
  pythonpkgs = pkgs.python310Packages.override {
    overrides = self: super: {

      scikit-gstat = super.buildPythonPackage rec {
        pname = "scikit-gstat";
        version = "1.0.8";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "5bHl6/otVWNk1r9K76KVGHLG3HgN7JcNOjqTg1jV414=";
        };
        propagatedBuildInputs = with super; [ numpy numba scipy pandas tqdm matplotlib imageio scikit-learn nose ];
        setuptoolsCheckPhase = "true";
      };

      geoutils = super.buildPythonPackage rec {
        pname = "geoutils";
        version = "0.0.9";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "9tq1Qoky6miSYOfO4iNMUXysvSW6CJ3gAkE1gCIrQkE=";
        };
        propagatedBuildInputs = with super; [
          geopandas
          tqdm
          matplotlib
          scipy
          rasterio
        ];
      };
      xdem = super.buildPythonPackage rec {
        pname = "xdem";
        version = "0.0.7";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "H+RuCeRoI3SsMITPPkhSagaJrR/+IPJPF0q/zwpPh4w=";
        };
        propagatedBuildInputs = with super; [
          opencv4
          scikit-learn
          scikitimage
          self.scikit-gstat
          self.geoutils
        ];
        setuptoolsCheckPhase = "true";
      };
    };
  };


in
(pythonpkgs.python.withPackages (_: with pythonpkgs; [
  ipython
  xarray
  h5netcdf
  odfpy
  pyarrow
  pytest
  rasterio
  xdem
  gdal
]
)).overrideAttrs (prev: {
  pname = "python";
})
