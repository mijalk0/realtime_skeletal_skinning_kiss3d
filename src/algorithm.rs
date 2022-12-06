use collada::document::ColladaDocument;
use kiss3d::{nalgebra as na, resource::Mesh};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::Path,
};

/// The geometry make up of an animated mesh.
pub struct Geometry {
    /// V
    pub vertices: Vec<na::Point3<f32>>,
    /// F
    pub faces: Vec<na::Point3<u16>>,
    /// W (V by 4 matrix of vertex bone weights)
    pub weights: Vec<na::Vector4<f32>>,
    /// J (V by 4 matrix of vertex bone indices)
    pub joints: Vec<na::Vector4<usize>>,
}

/// Holds all data in a `.dae` file.
pub struct Data {
    pub geometry: Geometry,
    skeleton: Vec<(usize, na::Matrix4<f32>)>,
    animation: Vec<Vec<(f32, na::Isometry3<f32>)>>,
}

impl Data {
    /// Computes the optimized Centers of Rotation, as specified in the paper.
    pub fn centers_of_rotation(
        &self,
        omega: f32,
        sigma: f32,
        epsilon: f32,
    ) -> Vec<Option<na::Vector3<f32>>> {
        let geometry = &self.geometry;

        let areas = tri_areas(geometry);

        // Need adjacency lists to speed up queries
        let vertex_adjacency_list = vertex_adjacency_list(geometry);
        let face_adjacency_list = face_adjacency_list(geometry);
        let face_face_adjacency_list = face_face_adjacency_list(geometry);

        // Par iter stands for parallel iterator.
        // This is fully multithreaded as in the paper
        (0..geometry.vertices.len())
            .into_par_iter()
            .map(|i| {
                let weight = weight_vector(geometry, i);

                let mut queue = VecDeque::from(vertex_adjacency_list[i].clone());
                let mut face_seeds = HashMap::new();
                let mut visited_vertices = HashSet::new();

                // Pass 1: Find all vertex seeds
                // Instead of approximate nearest neighbor search, I just perform a basic BFS search
                // from the vertex. This works fine due to the smooth skinning weights assumption.
                while !queue.is_empty() {
                    let curr_vertex = queue.pop_front().unwrap();
                    let dist = weight.dist(weight_vector(geometry, curr_vertex));

                    // Mark as visited
                    visited_vertices.insert(curr_vertex);
                    // If close enough:
                    if dist < omega {
                        // Add every face involving this vertex as a face seed
                        for face_index in face_adjacency_list[curr_vertex].iter() {
                            if !face_seeds.contains_key(face_index) {
                                let face = geometry.faces[*face_index];
                                let w_alpha = weight_vector(geometry, face[0] as usize);
                                let w_beta = weight_vector(geometry, face[1] as usize);
                                let w_gamma = weight_vector(geometry, face[2] as usize);

                                let mut w_alpha_beta_gamma = w_alpha.add(&(w_beta.add(&w_gamma)));
                                w_alpha_beta_gamma.mul_mut(1.0 / 3.0);

                                let similarity =
                                    similarity_function(&w_alpha_beta_gamma, &weight, sigma);

                                face_seeds.insert(
                                    face_index,
                                    Triangle {
                                        v_alpha: geometry.vertices[face[0] as usize],
                                        v_beta: geometry.vertices[face[1] as usize],
                                        v_gamma: geometry.vertices[face[2] as usize],

                                        a: areas[*face_index],

                                        similarity,
                                    },
                                );
                            }
                        }

                        // And we'll want to check its neighoburs
                        for second_neighbour in vertex_adjacency_list[curr_vertex].iter() {
                            // We haven't seen this yet
                            if !visited_vertices.contains(second_neighbour) {
                                visited_vertices.insert(*second_neighbour);
                                queue.push_back(*second_neighbour);
                            }
                        }
                    }
                }

                let mut visited_faces = face_seeds
                    .iter()
                    .map(|(face_index, _)| **face_index)
                    .collect::<HashSet<usize>>();

                let mut queue = VecDeque::new();
                for face_index in visited_faces.iter() {
                    for adjacent_face in face_face_adjacency_list[*face_index].iter() {
                        if !face_seeds.contains_key(adjacent_face) {
                            queue.push_back(adjacent_face);
                        }
                    }
                }

                // Pass 2: Find all face seeds
                // This is where we can apply the smooth skinning weights assumption, and walk the
                // triangle mesh adjacency graph
                while !queue.is_empty() {
                    let curr_face_index = queue.pop_front().unwrap();
                    let curr_face = geometry.faces[*curr_face_index];

                    let w_alpha = weight_vector(geometry, curr_face[0] as usize);
                    let w_beta = weight_vector(geometry, curr_face[1] as usize);
                    let w_gamma = weight_vector(geometry, curr_face[2] as usize);

                    let mut w_alpha_beta_gamma = w_alpha.add(&(w_beta.add(&w_gamma)));
                    w_alpha_beta_gamma.mul_mut(1.0 / 3.0);

                    let similarity = similarity_function(&w_alpha_beta_gamma, &weight, sigma);

                    let triangle = Triangle {
                        v_alpha: geometry.vertices[curr_face[0] as usize],
                        v_beta: geometry.vertices[curr_face[1] as usize],
                        v_gamma: geometry.vertices[curr_face[2] as usize],

                        a: areas[*curr_face_index],

                        similarity,
                    };

                    visited_faces.insert(*curr_face_index);
                    // Continue if large enough similarity
                    if similarity > epsilon {
                        face_seeds.insert(curr_face_index, triangle);

                        // And we'll want to check its neighbours
                        for second_neighbour in face_face_adjacency_list[*curr_face_index].iter() {
                            // We haven't seen this yet
                            if !visited_faces.contains(second_neighbour) {
                                visited_faces.insert(*second_neighbour);
                                queue.push_back(second_neighbour);
                            }
                        }
                    }
                }

                // At this point, weight holds the weight vector for the current vertex.
                // And similar_weights should hold all triangles with similar weights.

                let mut num = na::Vector3::zeros();
                let mut den = 0.0;

                for (_, triangle) in face_seeds.iter() {
                    let s = triangle.similarity;

                    let mut v_alpha_beta_gamma =
                        triangle.v_alpha.coords + triangle.v_beta.coords + triangle.v_gamma.coords;
                    v_alpha_beta_gamma /= 3.0;

                    let common = s * triangle.a;

                    num += common * v_alpha_beta_gamma;
                    den += common;
                }

                if den != 0.0 {
                    let result = num / den;
                    Some(result)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Simplistic sparse vector struct to handle niche cases with weight vectors.
#[derive(Clone, Debug)]
struct SparseVector {
    inner: HashMap<usize, f32>,
}

impl SparseVector {
    /// Interchangeable add/subtract function.
    fn add_or_sub(&self, other: &Self, add: bool) -> Self {
        let mut inner = HashMap::new();

        for (i, v1) in self.inner.iter() {
            if let Some(v2) = other.inner.get(i) {
                inner.insert(*i, v1 - v2);
            } else {
                inner.insert(*i, *v1);
            }
        }

        for (i, v) in &other.inner {
            if !self.inner.contains_key(&i) {
                if add {
                    inner.insert(*i, *v);
                } else {
                    inner.insert(*i, -*v);
                }
            }
        }

        Self { inner }
    }

    /// Add `SparseVector`s.
    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.add_or_sub(other, true)
    }

    /// Subtract `SparseVector`s.
    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.add_or_sub(other, false)
    }

    /// Multiply a `SparseVector` by a value in place.
    #[inline(always)]
    fn mul_mut(&mut self, mul: f32) {
        for (_, value) in self.inner.iter_mut() {
            *value *= mul;
        }
    }

    #[inline(always)]
    fn dist(&self, other: Self) -> f32 {
        let sub = self.sub(&other);

        sub.inner
            .iter()
            .map(|(_, v)| v)
            .fold(0.0, |tot, v| tot + v * v)
            .sqrt()
    }
}

/// The weight similarity function, as defined in the paper.
fn similarity_function(wp: &SparseVector, wv: &SparseVector, sigma: f32) -> f32 {
    let mut sum = 0.0;

    for (j, w_pj) in &wp.inner {
        for (k, w_vk) in &wv.inner {
            if j != k {
                let w_pk = *wp.inner.get(k).unwrap_or(&0.0);
                let w_vj = *wv.inner.get(j).unwrap_or(&0.0);

                let coeff = w_pj * w_pk * w_vj * w_vk;
                let exponent = -(w_pj * w_vk - w_pk * w_vj).powf(2.0) / sigma.powf(2.0);

                sum += coeff * exponent.exp();
            }
        }
    }

    sum
}

/// Constructs a sparse weight vector for vertex `i`.
#[inline(always)]
fn weight_vector(geometry: &Geometry, i: usize) -> SparseVector {
    let joints = geometry.joints[i];
    let weights = geometry.weights[i];

    let mut sparse_vec = SparseVector {
        inner: HashMap::with_capacity(4),
    };

    for (joint, weight) in joints.iter().zip(weights.iter()) {
        if let Some(val) = sparse_vec.inner.get_mut(joint) {
            *val += *weight;
        } else {
            sparse_vec.inner.insert(*joint, *weight);
        }
    }

    sparse_vec
}

/// Relevant triangle data required to precompute centers of rotation.
#[derive(Debug)]
struct Triangle {
    v_alpha: na::Point3<f32>,
    v_beta: na::Point3<f32>,
    v_gamma: na::Point3<f32>,

    a: f32,

    similarity: f32,
}

/// Builds the face-face mesh adjacency list (matrix), such that `F[i]` hold all the faces
/// involving face `i`.
fn face_face_adjacency_list(geometry: &Geometry) -> Vec<Vec<usize>> {
    let mut hashmap = HashMap::<(usize, usize), Vec<usize>>::new();

    geometry
        .faces
        .iter()
        .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
        .enumerate()
        .for_each(|(face_index, f)| {
            for i in 0..3 {
                for j in (i + 1)..3 {
                    if f[i] > f[j] {
                        if let Some(entry) = hashmap.get_mut(&(f[j], f[i])) {
                            entry.push(face_index);
                        } else {
                            let entry = vec![face_index];
                            hashmap.insert((f[j], f[i]), entry);
                        }
                    } else {
                        if let Some(entry) = hashmap.get_mut(&(f[i], f[j])) {
                            entry.push(face_index);
                        } else {
                            let entry = vec![face_index];
                            hashmap.insert((f[i], f[j]), entry);
                        }
                    }
                }
            }
        });

    let mut facemap = HashMap::<usize, HashSet<usize>>::new();

    for (face_index, f) in geometry.faces.iter().enumerate() {
        facemap.insert(face_index, HashSet::new());

        for i in 0..3 {
            for j in (i + 1)..3 {
                let e = if f[i] > f[j] {
                    (f[j] as usize, f[i] as usize)
                } else {
                    (f[i] as usize, f[j] as usize)
                };

                let entry = facemap.get_mut(&face_index).unwrap();

                for adjacent_face in hashmap.get(&e).unwrap() {
                    entry.insert(*adjacent_face);
                }
            }
        }
    }

    let mut vec = facemap
        .into_iter()
        .collect::<Vec<(usize, HashSet<usize>)>>();
    vec.sort_by_key(|v| v.0);

    vec.into_iter()
        .map(|v| v.1)
        .collect::<Vec<HashSet<usize>>>()
        .into_iter()
        .map(|hashset| hashset.into_iter().collect())
        .collect()
}

/// Builds the face triangle mesh adjacency list (matrix), such that `F[i]` hold all the faces
/// involving vertex `i`.
fn face_adjacency_list(geometry: &Geometry) -> Vec<Vec<usize>> {
    let mut hashmap = HashMap::<usize, Vec<usize>>::new();

    geometry
        .faces
        .iter()
        .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
        .enumerate()
        .for_each(|(face_index, face)| {
            for vertex_index in face {
                if let Some(entry) = hashmap.get_mut(&vertex_index) {
                    entry.push(face_index);
                } else {
                    let entry = vec![face_index];
                    hashmap.insert(vertex_index, entry);
                }
            }
        });

    let mut vec = hashmap.into_iter().collect::<Vec<(usize, Vec<usize>)>>();
    vec.sort_by_key(|v| v.0);
    vec.into_iter().map(|v| v.1).collect()
}

/// Builds the triangle mesh adjacency list (matrix), such that `V[i]` hold all the vertices which
/// are adjacent to vertex `i`.
fn vertex_adjacency_list(geometry: &Geometry) -> Vec<Vec<usize>> {
    let mut hashmap = HashMap::<usize, HashSet<usize>>::new();

    geometry
        .faces
        .iter()
        .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
        .for_each(|face| {
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let opposite = if i == 0 && j == 1 {
                        2
                    } else if i == 0 && j == 2 {
                        1
                    } else {
                        0
                    };

                    if let Some(entry) = hashmap.get_mut(&(face[opposite])) {
                        entry.insert(face[i]);
                        entry.insert(face[j]);
                    } else {
                        let mut entry = HashSet::new();
                        entry.insert(face[i]);
                        entry.insert(face[j]);
                        hashmap.insert(face[opposite], entry);
                    }
                }
            }
        });

    let mut vec = hashmap
        .into_iter()
        .collect::<Vec<(usize, HashSet<usize>)>>();
    vec.sort_by_key(|v| v.0);

    let out = vec
        .into_iter()
        .map(|v| v.1)
        .collect::<Vec<HashSet<usize>>>()
        .into_iter()
        .map(|hashset| hashset.into_iter().collect())
        .collect();

    out
}

/// Computes all triangle areas.
fn tri_areas(geometry: &Geometry) -> Vec<f32> {
    let mut a = Vec::new();
    for face in geometry.faces.iter() {
        let p1 = geometry.vertices[face[0] as usize];
        let p2 = geometry.vertices[face[1] as usize];
        let p3 = geometry.vertices[face[2] as usize];

        a.push(0.5 * (p1 - p3).cross(&(p2 - p3)).norm())
    }

    a
}

/// Which animation algorithm to animate a character with.
pub enum AnimAlgorithm<'a> {
    /// Use Linear Blend Skinning.
    LinearBlendSkinning,
    /// Use Real Time Skeletal Skinning. Holds optimized Centers of Rotation as well.
    RealTimeSkeletalSkinning(&'a Vec<Option<na::Vector3<f32>>>),
}

impl Data {
    #[inline(always)]
    pub fn animate(
        &self,
        t: f32,
        mode: AnimAlgorithm,
        scene_node: &mut kiss3d::scene::SceneNode,
        //window: &mut kiss3d::window::Window,
    ) {
        // Assume all channels have same length
        let len = self
            .animation
            .iter()
            .max_by(|x, y| {
                x.last()
                    .unwrap()
                    .0
                    .partial_cmp(&y.last().unwrap().0)
                    .unwrap()
            })
            .unwrap()
            .last()
            .unwrap()
            .0;

        // Wrap around time
        let t = if len == 0.0 || t == 0.0 {
            0.0
        } else {
            t.rem_euclid(len)
        };

        let local_poses: Vec<_> = self
            .animation
            .iter()
            .map(|key_frames| {
                let key_frame = key_frames
                    .iter()
                    .position(|(time, _)| t < *time)
                    .unwrap_or(key_frames.len() - 1);

                if key_frames.len() == 1 {
                    key_frames[0].1
                } else {
                    let t1 = key_frames[key_frame - 1].0;
                    let t2 = key_frames[key_frame].0;

                    let factor = (t - t1) / (t2 - t1);

                    let pos1 = key_frames[key_frame - 1].1;
                    let pos2 = key_frames[key_frame].1;

                    let pos = pos1.lerp_slerp(&pos2, factor);
                    pos
                }
            })
            .collect();

        let mut global_poses = Vec::new();

        // Initialize junk
        for _ in 0..local_poses.len() {
            global_poses.push(na::Isometry3::identity());
        }

        // So we can now use it
        for (i, (local_pose, (parent_index, _))) in
            local_poses.iter().zip(self.skeleton.iter()).enumerate()
        {
            if *parent_index == collada::ROOT_JOINT_PARENT_INDEX as usize {
                global_poses[i] = *local_pose;
            } else {
                global_poses[i] = global_poses[*parent_index] * *local_pose;
            }
        }

        // And finally we must apply the inverse bind poses to get the final joint matrices
        let joint_matrices: Vec<na::Matrix4<f32>> = global_poses
            .into_iter()
            .zip(self.skeleton.iter())
            .map(|(global_pose, (_, inverse_bind_pose))| {
                global_pose.to_homogeneous() * inverse_bind_pose
            })
            .collect();

        // To finish, we need to apply the animations to the vertices
        let mut v = self
            .geometry
            .vertices
            .iter()
            .map(|_| na::Point3::new(0.0, 0.0, 0.0))
            .collect::<Vec<_>>();

        match mode {
            // Run the Linear Blend Skinning Algorithm
            AnimAlgorithm::LinearBlendSkinning => {
                for (vertex_animated, (vertex_base, (weights, joints))) in v.iter_mut().zip(
                    self.geometry.vertices.iter().zip(
                        self.geometry
                            .weights
                            .iter()
                            .zip(self.geometry.joints.iter()),
                    ),
                ) {
                    let vertex_base =
                        na::Vector4::new(vertex_base.x, vertex_base.y, vertex_base.z, 1.0);

                    for i in 0..4 {
                        *vertex_animated +=
                            (weights[i] * joint_matrices[joints[i]] * vertex_base).xyz();
                    }
                }
            }
            // Run the Real Time Skeletal Skinning Algorithm
            AnimAlgorithm::RealTimeSkeletalSkinning(cor) => {
                let joint_quat = |i| {
                    let matrix: na::Matrix4<f32> = joint_matrices[i];
                    let q_matrix: na::Matrix3<f32> = matrix.fixed_slice::<3, 3>(0, 0).into_owned();
                    let q = *na::UnitQuaternion::from_matrix(&q_matrix).quaternion();

                    q
                };

                //                for ((vertex_animated, p_star_i), (vertex_base, (weights, joints))) in
                //                    v.iter_mut().zip(cor.iter()).zip(
                //                        self.geometry.vertices.iter().zip(
                //                            self.geometry
                //                                .weights
                //                                .iter()
                //                                .zip(self.geometry.joints.iter()),
                //                        ),
                //                    )
                //                {
                v.par_iter_mut()
                    .zip(cor.par_iter())
                    .zip(
                        self.geometry.vertices.par_iter().zip(
                            self.geometry
                                .weights
                                .par_iter()
                                .zip(self.geometry.joints.par_iter()),
                        ),
                    )
                    .for_each(
                        |((vertex_animated, p_star_i), (vertex_base, (weights, joints)))| {
                            let mut lbs_mat: na::Matrix4<f32> = na::Matrix4::zeros();

                            for i in 0..4 {
                                lbs_mat += weights[i] * joint_matrices[joints[i]];
                            }

                            if let Some(p_star_i) = p_star_i {
                                let q1 = joint_quat(joints[0]);
                                let mut q2 = joint_quat(joints[1]);
                                let mut q3 = joint_quat(joints[2]);
                                let mut q4 = joint_quat(joints[3]);

                                if q1.dot(&q2) < 0.0 {
                                    q2 = -q2;
                                }

                                if q1.dot(&q3) < 0.0 {
                                    q3 = -q3;
                                }

                                if q1.dot(&q4) < 0.0 {
                                    q4 = -q4;
                                }

                                let q = na::UnitQuaternion::new_normalize(
                                    weights[0] * q1
                                        + weights[1] * q2
                                        + weights[2] * q3
                                        + weights[3] * q4,
                                );
                                let r = q.to_rotation_matrix();

                                let r_tilde: na::Matrix3<f32> =
                                    lbs_mat.fixed_slice::<3, 3>(0, 0).into_owned();
                                let t_tilde: na::Vector3<f32> =
                                    lbs_mat.fixed_slice::<3, 1>(0, 3).into_owned();

                                let t = (r_tilde * p_star_i) + t_tilde - (r * p_star_i);
                                //points.push((offset * na::Vector4::new(t.x, t.y, t.z, 1.0)).xyz());
                                *vertex_animated = (r * vertex_base.coords + t).into();
                            } else {
                                let vertex_base = na::Vector4::new(
                                    vertex_base.x,
                                    vertex_base.y,
                                    vertex_base.z,
                                    1.0,
                                );

                                *vertex_animated = na::Point3::from((lbs_mat * vertex_base).xyz());
                            }
                        },
                    )
                // if draw_points {
                //     for point in points {
                //         window.draw_point(&point.into(), &na::Point3::new(1.0, 1.0, 1.0));
                //     }
                // }
            }
        }

        let mut data_mut = scene_node.data_mut();
        data_mut.modify_vertices(&mut |old_v: &mut Vec<na::Point3<f32>>| *old_v = v.clone());
        data_mut.recompute_normals();
    }
}

/// Load vertices of a collada document `Object`.
fn load_vertices(object: &collada::Object) -> Vec<na::Point3<f32>> {
    // Commonly written as V in assignments
    // This is like std::vector<Eigen::Vector3d>> in C++
    object
        .vertices
        .iter()
        .map(|v| na::Point3::new(v.x as f32, v.y as f32, v.z as f32))
        .collect()
}

/// Loads vertex weights and joints of a collada `Object`.
fn load_weights_joints(
    object: &collada::Object,
) -> (Vec<na::Vector4<f32>>, Vec<na::Vector4<usize>>) {
    let mut weights = Vec::new();
    let mut joints = Vec::new();

    for joint_weight in &object.joint_weights {
        // Collada doesn't require all weights to sum to 1, so I enforce it here.
        let w = na::Vector4::from(joint_weight.weights);
        let w = w / w.sum();

        weights.push(w);
        joints.push(na::Vector4::from(joint_weight.joints));
    }

    (weights, joints)
}

/// Load faces of a collada `Object`.
fn load_faces(object: &collada::Object) -> Result<Vec<na::Point3<u16>>, String> {
    let mut faces: Vec<na::Point3<u16>> = Vec::new();

    for geometry in &object.geometry {
        for mesh in &geometry.mesh {
            match mesh {
                collada::PrimitiveElement::Polylist(polylist) => {
                    for shape in &polylist.shapes {
                        match shape {
                            collada::Shape::Triangle(i1, i2, i3) => {
                                let i1 = i1.0 as u16;
                                let i2 = i2.0 as u16;
                                let i3 = i3.0 as u16;

                                faces.push(na::Point3::new(i1, i2, i3));
                            }
                            _ => return Err("don't support nontriangular shapes".to_string()),
                        }
                    }
                }
                collada::PrimitiveElement::Triangles(triangles) => {
                    for vertex in &triangles.vertices {
                        let i1 = vertex.0 as u16;
                        let i2 = vertex.1 as u16;
                        let i3 = vertex.2 as u16;

                        faces.push(na::Point3::new(i1, i2, i3));
                    }
                }
            }
        }
    }

    Ok(faces)
}

/// Creates `nalgebra`'s `Isometry3` which consists of a `UnitQuaternion` and `Translation` from a collada `Matrix4`.
fn matrix_conv(matrix: &collada::Matrix4<f32>) -> na::Isometry3<f32> {
    // Collada has everything in row major
    let matrix = na::Matrix4::from(*matrix).transpose();

    let r_mat = matrix.fixed_slice::<3, 3>(0, 0).into_owned();
    let r = na::UnitQuaternion::from_matrix(&r_mat);

    let t_mat = matrix.fixed_slice::<3, 1>(0, 3).into_owned();
    let t = na::Vector3::from(t_mat);

    na::Isometry3::from_parts(t.into(), r)
}

/// Creates a `kiss3d` `Mesh` as well as `Data` from a `.dae` file.
pub fn load_collada(path: impl AsRef<Path>) -> Result<(Mesh, Data), String> {
    let document = ColladaDocument::from_path(path.as_ref()).map_err(|err| err.to_string())?;
    let objects = document
        .get_obj_set()
        .ok_or("failed to load object set".to_string())?
        .objects;
    let object = objects.get(0).ok_or("ojbect set is empty".to_string())?;

    let vertices = load_vertices(&object);
    let faces = load_faces(&object)?;
    let (weights, joints) = load_weights_joints(&object);

    let skeletons = document
        .get_skeletons()
        .ok_or("file doesn't contain skeletons")?;

    let collada_skeleton = skeletons.get(0).ok_or("skeleton list is empty")?;

    let mut joint_map = HashMap::new();
    for (i, joint) in collada_skeleton.joints.iter().enumerate() {
        joint_map.insert(joint.name.clone(), i);
    }

    let skeleton: Vec<_> = collada_skeleton
        .joints
        .iter()
        .map(|joint| {
            (
                joint.parent_index as usize,
                // Collada has everything in row major
                na::Matrix4::from(joint.inverse_bind_pose).transpose(),
            )
        })
        .collect();

    let animations = document
        .get_animations()
        .ok_or("file doesn't contain animations")?;

    // Animations which will be sorted by joint index
    let mut sorted_anims = Vec::new();

    // Add animations for stationary joints
    for (i, joint) in collada_skeleton.joints.iter().enumerate() {
        if let Some(animation) = animations
            .iter()
            .find(|animation| animation.target == format!("{}/matrix", joint.name))
        {
            // Can't fail, we matched the suffix
            let name = animation.target.strip_suffix("/matrix").unwrap();
            let index = joint_map
                .get(name)
                .ok_or("animation uses unknown joint".to_string())?;

            sorted_anims.push((
                *index,
                animation.sample_times.clone(),
                animation.sample_poses.clone(),
            ));
        } else {
            // Add animations for stationary joints
            sorted_anims.push((i, vec![0.0], vec![collada_skeleton.bind_poses[i]]));
        }
    }

    let animation = sorted_anims
        .into_iter()
        .map(|anim| {
            // Used to offset all time, and to make animation loopable
            let t = anim.1[0];

            anim.1
                .iter()
                .map(|time| time - t)
                .zip(anim.2.into_iter().map(|matrix| matrix_conv(&matrix)))
                .collect()
        })
        .collect();

    let mesh = Mesh::new(vertices.clone(), faces.clone(), None, None, true);

    let geometry = Geometry {
        vertices,
        faces,
        weights,
        joints,
    };
    let data = Data {
        geometry,
        skeleton,
        animation,
    };

    Ok((mesh, data))
}
