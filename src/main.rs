#![feature(slice_group_by)]
#![feature(array_zip)]

mod accelerators;
mod consts;

use std::env;
use tobj;
use accelerators::bvh::build_bvh;
use consts::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];
    let scene = tobj::load_obj(
        filename,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    );

    let (models, _materials) = scene.expect("Failed to load OBJ file");

    let tris = models[0]
        .mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as Index, t[1] as Index, t[2] as Index])
        .collect();

    let get_verts_dim = |d: usize| models[0]
        .mesh
        .positions
        .iter()
        .copied()
        .skip(d)
        .step_by(3)
        .collect::<Vec<_>>();

    let verts = [get_verts_dim(0), get_verts_dim(1), get_verts_dim(2)];

    let _bvh = build_bvh(tris, verts);

    println!("Hello, world!");
}
