{
  description = "ROAST";

  inputs = {
    utils.url = "github:numtide/flake-utils";

    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    utils,
    nixpkgs,
    poetry2nix,
    pre-commit-hooks,
  }: let
    supportedSystems = ["x86_64-linux"];
  in
    utils.lib.eachSystem supportedSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          poetry2nix.overlay
          (_: old: {
            hdf5 = old.hdf5.override {mpiSupport = true;};
          })
        ];
      };

      inherit (pkgs) lib;

      projectDir = ./.;
      overrides = pkgs.poetry2nix.overrides.withDefaults (_: super: {
        # Prefer wheel for problematic packages
        furo = super.furo.override {preferWheel = true;};
        ruff = super.ruff.override {preferWheel = true;};
      });
      pythonEnv = pkgs.poetry2nix.mkPoetryEnv {
        inherit projectDir overrides;
        editablePackageSources.roast = ./src;
        groups = ["dev" "docs"];
      };
    in rec {
      checks = {
        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;

          hooks = {
            make_readme = {
              enable = true;
              name = "make-readme";
              entry = "${pkgs.stdenv.shell} .github/make-readme.sh";
              files = "README\.md";
              language = "system";
              pass_filenames = false;
            };

            alejandra.enable = true;
            black.enable = true;
            commitizen.enable = true;
            deadnix.enable = true;
            editorconfig-checker.enable = true;
            ruff.enable = true;
            statix.enable = true;
            taplo.enable = true;
            typos.enable = true;
          };
        };
      };

      devShells = {
        default = pythonEnv.env.overrideAttrs (old: {
          name = "roast-dev";
          nativeBuildInputs = with pkgs;
            old.nativeBuildInputs
            ++ [
              mpi
              poetry
            ];
          inherit (self.checks.${system}.pre-commit-check) shellHook;
        });
      };

      packages = {
        default = pkgs.poetry2nix.mkPoetryApplication {
          inherit projectDir;
        };

        roastDocHTML = pkgs.stdenvNoCC.mkDerivation rec {
          name = "roast-manual-html";
          src = self;
          buildInputs = [pythonEnv];
          buildPhase = ''
            export SOURCE_DATE_EPOCH=${builtins.toString self.sourceInfo.lastModified}
            cd docs
            make html
          '';
          installPhase = ''
            mkdir -p $out/share/doc/roast
            cp -r build/html $out/share/doc/roast
          '';
        };
      };
    });
}
