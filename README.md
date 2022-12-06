# Realtime Skeletal Skinning with `kiss3d`
Realtime Skeletal Skinning implemented in Rust, method from [Binh Huy Le & and Jessica K. Hodgins 2016](https://binh.graphics/papers/2016s-cor/)

# Preview
![image](https://github.com/mijalk0/realtime_skeletal_skinning_kiss3d/blob/master/preview.png)

The grey mannequin is rendered with Linear Blend Skinning. The yellow mannequin uses the Realtime Skeletal Skinning technique. As a result, it has less artifacts around the elbows and similar creases.

# Install Rust

Ensure `rustc` and `cargo` are installed. If running Unix/Linux/MacOS, issue

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

to install rust if needed. Check the [official rust installation guide](https://www.rust-lang.org/tools/install) to be sure.

# Building

To build the binary, clone the repository and `cd` into it. Then issue 

```
cargo build --release
```

# Running

To run the binary, make sure your current working directory is the repository. The binary needs this to parse the `.dae` files of the animations. Then simply issue

```
cargo run --release
```

To pass in arguments, add them after the `--release` flag, e.g.

```
cargo run --release file Swing\ Dancing.dae
```

to run with the `Swing Dancing.dae` animation.

# Uninstall Rust

To uninstall the rust toolchain, issue 

```
rustup self uninstall
```
