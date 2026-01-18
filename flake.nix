{
  description = "Kokoros - Rust TTS workspace";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true; # Required for CUDA
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # ONNX Runtime with CUDA support
        onnxruntimeCuda = pkgs.onnxruntime.override {
          cudaSupport = true;
        };

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          cmake
          llvmPackages.libclang
          gcc
          alsa-utils
        ];

        commonBuildInputs = with pkgs; [
          # espeak-ng for espeak-rs
          espeak-ng

          # Audio encoding libraries
          libopus
          libogg
          lame

          # OpenSSL for reqwest
          openssl
        ] ++ lib.optionals stdenv.isDarwin [
          apple-sdk_15
        ];

        buildInputs = commonBuildInputs ++ [ pkgs.onnxruntime ];
        buildInputsCuda = commonBuildInputs ++ [ onnxruntimeCuda ];

        commonEnv = {
          RUST_BACKTRACE = "1";
          PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" buildInputs;
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = builtins.concatStringsSep " " [
            "-I${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include"
            "-I${pkgs.glibc.dev}/include"
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs buildInputs;

          env = commonEnv // {
            ORT_STRATEGY = "system";
            ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
          };

          shellHook = ''
            echo "Kokoros development shell"
            echo "Rust: $(rustc --version)"
          '';
        };

        devShells.cuda = pkgs.mkShell {
          nativeBuildInputs = nativeBuildInputs;
          buildInputs = buildInputsCuda;

          env = commonEnv // {
            ORT_STRATEGY = "system";
            ORT_LIB_LOCATION = "${onnxruntimeCuda}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
            # Add CUDA to library path for runtime
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              onnxruntimeCuda
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
            ];
          };

          shellHook = ''
            echo "Kokoros development shell (CUDA enabled)"
            echo "Rust: $(rustc --version)"
            echo "CUDA: $(nvcc --version 2>/dev/null || echo 'not in PATH')"
          '';
        };
      }
    );
}
