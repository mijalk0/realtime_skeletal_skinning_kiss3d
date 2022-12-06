mod algorithm;

const CHARACTER_SCALE: f32 = 0.005;
const CHARACTER_LBS_OFFSET: na::Translation3<f32> = na::Translation3::new(-0.5, -0.5, 0.0);
const CHARACTER_REALTIMESKELETALSKINNING_OFFSET: na::Translation3<f32> =
    na::Translation3::new(0.5, -0.5, 0.0);

use kiss3d::light::Light;
use kiss3d::nalgebra as na;
use kiss3d::window::Window;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let mut window = Window::new("Real Time Skeletal Skinning");

    window.set_light(Light::StickToCamera);
    window.set_background_color(0.4, 0.3, 0.6);
    let file = get_file();
    let (mesh1, data1) = algorithm::load_collada(&file).expect(".dae file failed to load");
    let (mesh2, data2) = algorithm::load_collada(&file).expect(".dae file failed to load");

    // Used in paper
    const OMEGA: f32 = 0.1;
    const SIGMA: f32 = 0.1;
    const EPSILON: f32 = 10e-6;

    let cor = data2.centers_of_rotation(OMEGA, SIGMA, EPSILON);
    let mesh1 = Rc::new(RefCell::new(mesh1));
    let mesh2 = Rc::new(RefCell::new(mesh2));

    window.set_point_size(3.0);

    let mut m1 = window.add_mesh(mesh1, na::Vector3::from_element(CHARACTER_SCALE));
    //let mut m2 = window.add_mesh(mesh2, na::Vector3::from_element(1.0));
    let mut m2 = window.add_mesh(mesh2, na::Vector3::from_element(CHARACTER_SCALE));

    m1.set_color(1.0, 1.0, 1.0);
    m1.prepend_to_local_translation(&CHARACTER_LBS_OFFSET);

    m2.set_color(1.0, 1.0, 0.0);
    m2.prepend_to_local_translation(&CHARACTER_REALTIMESKELETALSKINNING_OFFSET);

    let mut now = std::time::Instant::now();
    let mut t = 0.0;
    let mut pause = false;
    let mut draw_points = true;

    println!("---------------=============COMMANDS=============---------------");
    println!("To load a different file, rerun with: cargo run --release -- file {{filename}}");
    println!("Space to pause the animation");
    println!("R to restart the animation");

    while window.render() {
        for event in window.events().iter() {
            match event.value {
                kiss3d::event::WindowEvent::Key(key, kiss3d::event::Action::Press, _modifiers) => {
                    match key {
                        kiss3d::event::Key::Space => pause = !pause,
                        kiss3d::event::Key::R => t = 0.0,
                        kiss3d::event::Key::P => draw_points = !draw_points,
                        _ => (),
                    }
                }
                _ => (),
            }
        }

        let tmp = std::time::Instant::now();
        let dt = if !pause {
            (tmp - now).as_secs_f32()
        } else {
            0.0
        };
        now = tmp;
        t += dt;
        data1.animate(
            t,
            algorithm::AnimAlgorithm::LinearBlendSkinning,
            &mut m1,
            // &mut window,
            // CHARACTER_LBS_OFFSET
            //     .to_homogeneous()
            //     .append_scaling(CHARACTER_SCALE),
            // draw_points,
        );
        data2.animate(
            t,
            algorithm::AnimAlgorithm::RealTimeSkeletalSkinning(&cor),
            &mut m2,
            // &mut window,
            // CHARACTER_REALTIMESKELETALSKINNING_OFFSET
            //     .to_homogeneous()
            //     .append_scaling(CHARACTER_SCALE),
            // draw_points,
        );
    }
}

fn get_file() -> String {
    if let Some(i) = std::env::args().position(|x| x == "file") {
        std::env::args()
            .collect::<Vec<String>>()
            .get(i + 1)
            .unwrap()
            .clone()
    } else {
        "Dancing.dae".to_string()
    }
}
