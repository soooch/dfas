#![feature(array_zip)]
//#![feature(portable_simd)]
#![feature(array_chunks)]
use std::thread;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use packed_simd::*;
use rayon::prelude::*;
pub type Real = f32;
pub type Index = usize;
pub const MAX_TRIS_PER_NODE: usize = 20;

fn get_min_bounds(tris: &[[Index; 3]], verts: &[Vec<Real>; 3]) -> [Vec<Real>; 3] {
    let get_min_bounds_for_dim = |d: Index| tris
        .iter()
        .map(|&[a, b, c]| verts[d][a].min(verts[d][b].min(verts[d][c])))
        .collect::<Vec<_>>();

    [ get_min_bounds_for_dim(0)
    , get_min_bounds_for_dim(1)
    , get_min_bounds_for_dim(2)
    ]
}

fn get_max_bounds(tris: &[[Index; 3]], verts: &[Vec<Real>; 3]) -> [Vec<Real>; 3] {
    let get_max_bounds_for_dim = |d: Index| tris
        .iter()
        .map(|&[a, b, c]| verts[d][a].max(verts[d][b].max(verts[d][c])))
        .collect::<Vec<_>>();

    [ get_max_bounds_for_dim(0)
    , get_max_bounds_for_dim(1)
    , get_max_bounds_for_dim(2)
    ]
}

fn get_centroids(min_bounds: &[Vec<Real>; 3], max_bounds: &[Vec<Real>; 3]) -> [Vec<Real>; 3] {
    let get_centroids_for_dim = |d: Index| min_bounds[d]
        .iter()
        .zip(max_bounds[d].iter())
        .map(|(min, max)| 0.5 * min + 0.5 * max)
        .collect::<Vec<_>>();

    [ get_centroids_for_dim(0)
    , get_centroids_for_dim(1)
    , get_centroids_for_dim(2)
    ]

}

fn get_scene_min_bound(min_bounds: &[Vec<Real>; 3]) -> [Real; 3] {
    let get_scene_min_bound_for_dim = |d: Index| min_bounds[d]
        .iter()
        .copied()
        .reduce(Real::min)
        .unwrap();

    [ get_scene_min_bound_for_dim(0)
    , get_scene_min_bound_for_dim(1)
    , get_scene_min_bound_for_dim(2)
    ]
}

fn get_scene_max_bound(max_bounds: &[Vec<Real>; 3]) -> [Real; 3] {
    let get_scene_max_bound_for_dim = |d: Index| max_bounds[d]
        .iter()
        .copied()
        .reduce(Real::max)
        .unwrap();

    [ get_scene_max_bound_for_dim(0)
    , get_scene_max_bound_for_dim(1)
    , get_scene_max_bound_for_dim(2)
    ]
}

fn step_3_dep(x: u32) -> u32 {
    // TODO: use pdep for intel bmi2
    // ripped from pbrtv3
    let mut m = (x | (x << 16)) & 0x30000ff;
    // m = ---- --98 ---- ---- ---- ---- 7654 3210
    m = (m | (m << 8)) & 0x300f00f;
    // m = ---- --98 ---- ---- 7654 ---- ---- 3210
    m = (m | (m << 4)) & 0x30c30c3;
    // m = ---- --98 ---- 76-- --54 ---- 32-- --10
    m = (m | (m << 2)) & 0x9249249;
    // m = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    m
}
fn step_3_dep_simd(x: u32x8) -> u32x8 {
    // TODO: use pdep for intel bmi2
    // ripped from pbrtv3
    let mut m = (x | (x << 16)) & 0x30000ff;
    // m = ---- --98 ---- ---- ---- ---- 7654 3210
    m = (m | (m << 8)) & 0x300f00f;
    // m = ---- --98 ---- ---- 7654 ---- ---- 3210
    m = (m | (m << 4)) & 0x30c30c3;
    // m = ---- --98 ---- 76-- --54 ---- 32-- --10
    m = (m | (m << 2)) & 0x9249249;
    // m = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    m
}

fn testa(points: &Vec<[f32; 3]>, min_bound: [f32; 3], range: [f32; 3]) -> Vec<(usize, u32)> {
    let cdivr = range.map(|r| 1024.0 / r);
    points
        .par_chunks(512)
        .map(|chunk| chunk
            .iter()
            .map(|[x, y, z]| [(x - min_bound[0]) * cdivr[0], (y - min_bound[1]) * cdivr[1], (z - min_bound[2]) * cdivr[2]])
            .map(|point| point.map(|x| x as u32))
            .map(|point| point.map(step_3_dep))
            .map(|[x, y, z]| [x << 2, y << 1, z])
            .map(|[x, y, z]| x | y | z)
            .enumerate())
        .flatten_iter()
        .collect()
}

pub fn testb(points: &[Vec<f32>; 3], min_bound: [f32; 3], range: [f32; 3]) -> Vec<(usize, u32)> {
    // TODO: simd this
    let cdivr = range.map(|r| 1024.0 / r);
    points[0]
        .array_chunks::<8>()
        .map(|&x| f32x8::from(x))
        .map(move |x| (x - min_bound[0]) * cdivr[0])
        .map(u32x8::from_cast)
        .map(step_3_dep_simd)
        .map(move |x| x << 2)
        .zip(points[1]
            .array_chunks::<8>()
            .map(|&y| f32x8::from(y))
            .map(move |y| (y - min_bound[1]) * cdivr[1])
            .map(u32x8::from_cast)
            .map(step_3_dep_simd)
            .map(move |y| y << 1))
        .zip(points[2]
            .array_chunks::<8>()
            .map(|&z| f32x8::from(z))
            .map(move |z| (z - min_bound[2]) * cdivr[2])
            .map(u32x8::from_cast)
            .map(step_3_dep_simd))
        .map(|((x, y), z)| x | y | z)
        .flat_map(|x| {let mut out = [0; 8]; x.write_to_slice_unaligned(&mut out); out})
        .enumerate()
        .collect::<Vec<_>>()
}

pub fn testc(points: &[Vec<f32>; 3], min_bound: [f32; 3], range: [f32; 3]) -> Vec<(usize, u32)> {
    let cdivr = range.map(|r| 1024.0 / r);
    let cdivr_splat = cdivr.map(|c| f32x8::splat(c));
    let min_bc_div_r = cdivr.zip(min_bound).map(|(cr, min)| -cr * min).map(|c| f32x8::splat(c));

    // TODO: remainder after simd chunks
    // TODO: consider always keeping items in simd vectors

    let to_or = |dim: usize| points[dim]
            .array_chunks::<8>()
            .map(|&x| f32x8::from(x))
            //.map(move |x| (x - min_bound[dim]) * cdivr[dim])
            .map(move |x| x.mul_adde(cdivr_splat[dim], min_bc_div_r[dim]))
            .map(u32x8::from_cast)
            .map(step_3_dep_simd)
            .map(move |x| x << (2 - dim as u32));

    to_or(0)
        .zip(to_or(1))
        .zip(to_or(2))
        .map(|((x, y), z)| x | y | z)
        .flat_map(|x| {let mut out = [0; 8]; x.write_to_slice_unaligned(&mut out); out})
        .enumerate()
        .collect()
}

pub fn testd(points: &[Vec<f32>; 3], min_bound: [f32; 3], range: [f32; 3]) -> Vec<(usize, u32)> {
    let cdivr = range.map(|r| 1024.0 / r);

    // TODO: remainder after simd chunks
    // TODO: consider always keeping items in simd vectors

    /*
    let dim_points = |dim: usize| points[dim]
        .par_chunks(512)
        .map(|chunk| chunk
            .array_chunks::<8>()
            .map(|&x| f32x8::from(x))
            .map(move |x| (x - min_bound[dim]) * cdivr[dim])
            .map(u32x8::from_cast)
            .map(step_3_dep_simd)
            .map(move |x| x << (2 - dim as u32)))
        .flatten_iter()
        .collect();
    */

    let points = |dim: usize| points[dim]
        .par_chunks(512)
        .map(|chunk| chunk
            .array_chunks::<8>()
            .map(|x| f32x8::from_slice_aligned(x))
            .map(move |x| (x - min_bound[dim]) * cdivr[dim])
            .map(u32x8::from_cast)
            .map(step_3_dep_simd)
            .map(move |x| x << (2 - dim as u32)))
        .flatten_iter()
        .collect::<Vec<_>>();

    points(0).into_par_iter()
        .zip_eq(points(1))
        .zip_eq(points(2))
        .map(|((x, y), z)| x | y | z)
        .enumerate()
        .flat_map(|(idx, x)| {let mut out = [0; 8]; x.write_to_slice_unaligned(&mut out); out.into_par_iter().enumerate().map(move |(i, e)| (idx * 8 + i, e))})
        .collect()
}

pub fn teste(points: &[Vec<f32>; 3], min_bound: [f32; 3], range: [f32; 3]) -> Vec<(usize, u32)> {
    const NUM_THREADS: usize = 3;
    const SIMD_WIDTH: usize = 8;

    let cdivr = range.map(|r| 1024.0 / r);
    let p_per_t = points[0].len() / (NUM_THREADS);
    let p_overflow = points[0].len() % (NUM_THREADS);

    (0..NUM_THREADS)
        .into_par_iter()
        .map(|tid| (tid * p_per_t, (tid + 1) * p_per_t))
        .flat_map_iter(|(start, end)| {
            let to_or = |dim: usize| points[dim][start..end]
                .array_chunks::<SIMD_WIDTH>()
                .map(|&x| f32x8::from(x))
                .map(move |x| (x - min_bound[dim]) * cdivr[dim])
                .map(u32x8::from_cast)
                .map(step_3_dep_simd)
                .map(move |x| x << (2 - dim as u32));

            to_or(0)
                .zip(to_or(1))
                .zip(to_or(2))
                .map(|((x, y), z)| x | y | z)
                .flat_map(|x| {let mut out = [0; 8]; x.write_to_slice_unaligned(&mut out); out})
                .enumerate()
                .map(move |(idx, mort)| (idx + start, mort))
        })
        .collect()
}


fn criterion_benchmark(c: &mut Criterion) {
    let filename = "xyzrgb_dragon.obj";
    let scene = tobj::load_obj(
        filename,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    );

    let (models, _materials) = scene.expect("Failed to load OBJ file");

    let tris: Vec<[Index; 3]> = models
        .iter()
        .map(|m| m.mesh
            .indices
            .chunks_exact(3)
            .map(|t| [t[0] as Index, t[1] as Index, t[2] as Index]))
        .flatten()
        .collect();


    let get_verts_dim = |d: usize| models
        .iter()
        .map(|m| m.mesh
            .positions
            .iter()
            .copied()
            .skip(d)
            .step_by(3))
        .flatten()
        .collect::<Vec<_>>();


    let verts = [get_verts_dim(0), get_verts_dim(1), get_verts_dim(2)];

    let tri_min_bounds = get_min_bounds(&tris, &verts);
    let tri_max_bounds = get_max_bounds(&tris, &verts);

    let centroids = get_centroids(&tri_min_bounds, &tri_max_bounds);

    let scene_min_bound = get_scene_min_bound(&tri_min_bounds);
    let scene_max_bound = get_scene_max_bound(&tri_max_bounds);

    let range = scene_max_bound.zip(scene_min_bound).map(|(max, min)| max - min);


    let c_t = centroids[0]
        .iter()
        .cloned()
        .zip(centroids[1].iter().cloned())
        .zip(centroids[2].iter().cloned())
        .map(|((x, y), z)| [x, y, z])
        .collect();


    /*
    let outa = testa(&c_t, scene_min_bound, range);
    let outc = testc(&centroids, scene_min_bound, range);
    for i in 0..15 {
        println!(
            "{}\n{:#010b}, {:#010b}, {:#010b}\n{:#032b}\n{:#032b}\n",
            i,
            (((c_t[i][0] - scene_min_bound[0]) / range[0]) * 1024.0) as u32,
            (((c_t[i][1] - scene_min_bound[1]) / range[1]) * 1024.0) as u32,
            (((c_t[i][2] - scene_min_bound[2]) / range[2]) * 1024.0) as u32,
            outa[i].1, outc[i].1
        );
    }
    */
    //assert_eq!(testa(&c_t, scene_min_bound, range), testb(&centroids, scene_min_bound, range));
    //assert_eq!(testa(&c_t, scene_min_bound, range), testc(&centroids, scene_min_bound, range));
    //assert_eq!(testa(&c_t, scene_min_bound, range), testd(&centroids, scene_min_bound, range));



    let mut group = c.benchmark_group("mort");


    group.bench_function("a", |b| b.iter(|| testa(black_box(&c_t), black_box(scene_min_bound), black_box(range))));
    group.bench_function("b", |b| b.iter(|| testb(black_box(&centroids), black_box(scene_min_bound), black_box(range))));
    group.bench_function("c", |b| b.iter(|| testc(black_box(&centroids), black_box(scene_min_bound), black_box(range))));
    //group.bench_function("d", |b| b.iter(|| testd(black_box(&centroids), black_box(scene_min_bound), black_box(range))));
    group.bench_function("e", |b| b.iter(|| teste(black_box(&centroids), black_box(scene_min_bound), black_box(range))));
    group.finish();
    //println!("{}", c_t.len());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
