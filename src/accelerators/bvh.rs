use std::iter::Zip;

use crate::consts::*;

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

fn get_leaf_min_bound(min_bounds: &[Vec<Real>; 3], tris: &[(usize, u32)], offset: Index, length: Index) -> [Real; 3] {
    let get_leaf_min_bounds_for_dim = |d: Index| tris
        .iter()
        .skip(offset)
        .take(length)
        .map(|&(idx, _m)| min_bounds[d][idx])
        .reduce(Real::min)
        .unwrap();
    
    [ get_leaf_min_bounds_for_dim(0)
    , get_leaf_min_bounds_for_dim(1)
    , get_leaf_min_bounds_for_dim(2)
    ]
}

fn get_leaf_max_bound(max_bounds: &[Vec<Real>; 3], tris: &[(usize, u32)], offset: Index, length: Index) -> [Real; 3] {
    let get_leaf_max_bounds_for_dim = |d: Index| tris
        .iter()
        .skip(offset)
        .take(length)
        .map(|&(idx, _m)| max_bounds[d][idx])
        .reduce(Real::max)
        .unwrap();

    [ get_leaf_max_bounds_for_dim(0)
    , get_leaf_max_bounds_for_dim(1)
    , get_leaf_max_bounds_for_dim(2)
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

fn _encode_morton(x: Real, y: Real, z: Real) -> u32 {
    assert!(x >= 0.0);
    assert!(y >= 0.0);
    assert!(z >= 0.0);
    (step_3_dep(x.to_bits()) << 2) | (step_3_dep(y.to_bits()) << 1) | step_3_dep(z.to_bits())
}

pub trait FlatZip: Iterator {
    fn flat_zip(self) -> 
    Zip<<<Self as Iterator>::Item as IntoIterator>::IntoIter, <<Self as Iterator>::Item as IntoIterator>::IntoIter> 
    where 
        Self: Sized, 
        Self::Item: Sized, 
        Self::Item: IntoIterator;
}

impl<I> FlatZip for I where I: Iterator {
    #[inline]
    fn flat_zip(self) -> Zip<<Self::Item as IntoIterator>::IntoIter, <Self::Item as IntoIterator>::IntoIter> 
    where 
        Self: Sized, 
        Self::Item: Sized, 
        Self::Item: IntoIterator,
    {
        //Zip::new(self.next, self.next)
        unimplemented!()
    }
}

fn get_mortons(points: &[Vec<Real>; 3], min_bound: [Real; 3], max_bound: [Real; 3]) -> Vec<(usize, u32)> {
    const MORTON_BITS: u32 = 10;
    const MORTON_SCALE: u32 = 1 << MORTON_BITS;

    let range = max_bound.zip(min_bound).map(|(max, min)| max - min);

    // doing stuff this way makes it easier for the compiler to batch operations into simd registers
    /*
    let mut mortons = points
        .iter()
        .enumerate()
        .map(|(dim, p_list)| p_list
            .iter()
            .map(move |x| ((x - min_bound[dim]) / range[dim]) * MORTON_SCALE as Real)
            .map(Real::to_bits)
            .map(step_3_dep)
            .map(move |x| x << (2 - dim)))
        .flat_zip()
        .map(|(x, y, z)| x | y | z)
        .enumerate()
        .collect::<Vec<_>>();
        */

    let fake_points: Vec<[Real; 3]> = Vec::new();
    let _fake_mort = fake_points
        .iter()
        .map(|[x, y, z]| {
            [
                ((x - min_bound[0]) / range[0]) * MORTON_SCALE as Real,
                ((y - min_bound[0]) / range[1]) * MORTON_SCALE as Real,
                ((z - min_bound[0]) / range[2]) * MORTON_SCALE as Real
            ]
        })
        .map(|[x, y, z]| [x.to_bits(), y.to_bits(), z.to_bits()])
        .map(|[x, y, z]| [step_3_dep(x), step_3_dep(y), step_3_dep(z)])
        .map(|[x, y, z]| [x << 2, y << 1, z])
        .map(|[x, y, z]| x | y | z)
        .enumerate()
        .collect::<Vec<_>>();

    let _fake_mort2 = fake_points
        .iter()
        .map(|point| point.map(|x| ((x - min_bound[0]) / range[0]) * MORTON_SCALE as Real))
        .map(|point| point.map(Real::to_bits))
        .map(|point| point.map(step_3_dep))
        .map(|[x, y, z]| [x << 2, y << 1, z])
        .map(|[x, y, z]| x | y | z)
        .enumerate()
        .collect::<Vec<_>>();

    let mut mortons = points[0]
        .iter()
        .map(|x| ((x - min_bound[0]) / range[0]) * MORTON_SCALE as Real)
        .map(|x| x as u32)
        .map(step_3_dep)
        .map(|x| x << 2)
        .zip(points[1]
            .iter()
            .map(|y| ((y - min_bound[1]) / range[1]) * MORTON_SCALE as Real)
            .map(|y| y as u32)
            .map(step_3_dep)
            .map(|y| y << 1))
        .zip(points[2]
            .iter()
            .map(|z| ((z - min_bound[2]) / range[2]) * MORTON_SCALE as Real)
            .map(|z| z as u32)
            .map(step_3_dep))
        .map(|((x, y), z)| x | y | z)
        .enumerate()
        .collect::<Vec<_>>();
    
    // TODO: switch to radix sort (rdxsort?)
    mortons.sort_by_key(|&(_, m)| m);

    mortons
}

//enum Axis

fn emit_lbvh
(_offset: Index, treelet_tris: &[(usize, u32)], min_bounds: &[Vec<Real>; 3], max_bounds: &[Vec<Real>; 3])
-> 
([[Vec<Real>; 3]; 2], Vec<u8>, Vec<[Index; 2]>) {
    const FIRST_BIT_INDEX: i32 = 29 - 12;
    let num_leafs = (treelet_tris.len() + MAX_TRIS_PER_NODE - 1) / MAX_TRIS_PER_NODE;
    let num_inner = num_leafs.next_power_of_two() - 1;
    let num_bounds = num_inner + num_leafs;
    let mut bounds = [ [ vec![Default::default(); num_bounds]
                       , vec![Default::default(); num_bounds]
                       , vec![Default::default(); num_bounds]
                       ]
                     , [ vec![Default::default(); num_bounds]
                       , vec![Default::default(); num_bounds]
                       , vec![Default::default(); num_bounds]
                       ]
                     ];
    let mut leafs = vec![[0, 0]; num_leafs];
    let inner = vec![3; num_inner];
    let mut stack = vec![(FIRST_BIT_INDEX, 0, treelet_tris.len(), 0)];

    while !stack.is_empty() {
        let (bit_idx, offset, num_tris, bounds_idx) = stack.pop().unwrap();
        if bit_idx == -1 || num_tris < MAX_TRIS_PER_NODE {
            // TODO: maybe change data arrangement?
            // TODO: might also want to just do the function for each store
            let leaf_min_bound = get_leaf_min_bound(min_bounds, treelet_tris, offset, num_tris);
            let leaf_max_bound = get_leaf_max_bound(max_bounds, treelet_tris, offset, num_tris);
            // TODO: fix indexing for bounds and leafs.
            // 2*p_idx+(1 or 2)
            bounds[0][0][bounds_idx] = leaf_min_bound[0];
            bounds[0][1][bounds_idx] = leaf_min_bound[1];
            bounds[0][2][bounds_idx] = leaf_min_bound[2];
            bounds[1][0][bounds_idx] = leaf_max_bound[0];
            bounds[1][1][bounds_idx] = leaf_max_bound[1];
            bounds[1][2][bounds_idx] = leaf_max_bound[2];
            leafs[bounds_idx - num_inner] = [offset, num_tris];
        }
        else {
            let mask = 1 << bit_idx;
            if treelet_tris[0].1 & mask == treelet_tris[num_tris - 1].1 & mask {
                stack.push((bit_idx - 1, offset, num_tris, bounds_idx));
            }
            else {
                
            }
        }
    }

    (bounds, inner, leafs)
}

pub fn build_bvh(tris: Vec<[Index; 3]>, verts: [Vec<Real>; 3]) -> Vec<Index> {
    let bvh = Vec::with_capacity(2 * tris.len() - 1);

    // TODO: make sure compiler understands all vectors are same len
    // .size_hint() maybe?
    let tri_min_bounds = get_min_bounds(&tris, &verts);
    let tri_max_bounds = get_max_bounds(&tris, &verts);
    
    let centroids = get_centroids(&tri_min_bounds, &tri_max_bounds);

    let scene_min_bound = get_scene_min_bound(&tri_min_bounds);
    let scene_max_bound = get_scene_max_bound(&tri_max_bounds);
    
    const MASK_TOP_12: u32 = 0x3ffc0000;
    let mortons = get_mortons(&centroids, scene_min_bound, scene_max_bound);
    let _treelets = mortons
        .group_by(|&(_, a), &(_, b)| a & MASK_TOP_12 == b & MASK_TOP_12)
        .scan(0, |state, t| {
            let offset = *state;
            *state += t.len();
            Some((offset, t))
        })
        .map(|(offset, tree)| emit_lbvh(offset, tree, &tri_min_bounds, &tri_max_bounds))
        .collect::<Vec<_>>();
    
    bvh
}

fn intersect(_o: [Real; 3], _d: [Real; 3], _dinv: [Real; 3], _bmin: [Real; 3], _bmax: [Real; 3]) -> bool {
    // TODO: batch rays and AABBs together then vector process?
    false
}