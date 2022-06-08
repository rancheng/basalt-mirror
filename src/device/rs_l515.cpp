/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko, Michael Loipführer and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/device/rs_l515.h>

std::string get_date();

namespace basalt {

RsL515Device::RsL515Device(bool manual_exposure, int skip_frames,
                           int webp_quality, double exposure_value)
    : manual_exposure(manual_exposure),
      skip_frames(skip_frames),
      webp_quality(webp_quality) {
  rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
  pipe = rs2::pipeline(context);

  config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
  config.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_Y16);
//  config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);

  if (context.query_devices().size() == 0) {
    std::cout << "Waiting for device to be connected" << std::endl;
    rs2::device_hub hub(context);
    hub.wait_for_device();
  }

  for (auto& s : context.query_devices()[0].query_sensors()) {
    std::cout << "Sensor " << s.get_info(RS2_CAMERA_INFO_NAME)
              << ". Supported options:" << std::endl;

    for (const auto& o : s.get_supported_options()) {
      std::cout << "\t" << rs2_option_to_string(o) << std::endl;
    }
  }

  auto device = context.query_devices()[0];
  // device.hardware_reset();

  std::cout << "Device " << device.get_info(RS2_CAMERA_INFO_NAME)
            << " connected" << std::endl;
  // device.query_sensors()[0] is depth camera
  sensor = device.query_sensors()[1];

  if (manual_exposure) {
    std::cout << "Enabling manual exposure control" << std::endl;
    if (sensor.supports(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
      sensor.set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
      sensor.set_option(rs2_option::RS2_OPTION_EXPOSURE, exposure_value * 1000);
    } else {
      std::cout << "Auto Exposure not supported!" << std::endl;
    }
  }
//  profile = pipe.start(config);
//  pipe.stop();
//  align_to =
//      RsL515Device::find_stream_to_align(profile.get_streams());
//  align = new rs2::align(align_to);
}

void RsL515Device::start() {
  auto callback = [&](const rs2::frame& frame) {
    exportCalibration();

    if (auto fp = frame.as<rs2::motion_frame>()) {
      auto motion = frame.as<rs2::motion_frame>();

      if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO &&
          motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
        RsIMUData d;
        d.timestamp = motion.get_timestamp();
        d.data << motion.get_motion_data().x, motion.get_motion_data().y,
            motion.get_motion_data().z;

        gyro_data_queue.emplace_back(d);
      } else if (motion &&
                 motion.get_profile().stream_type() == RS2_STREAM_ACCEL &&
                 motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
        RsIMUData d;
        d.timestamp = motion.get_timestamp();
        d.data << motion.get_motion_data().x, motion.get_motion_data().y,
            motion.get_motion_data().z;

        if (!prev_accel_data.get()) {
          prev_accel_data.reset(new RsIMUData(d));
        } else {
          BASALT_ASSERT(d.timestamp > prev_accel_data->timestamp);

          while (!gyro_data_queue.empty() && gyro_data_queue.front().timestamp <
                                                 prev_accel_data->timestamp) {
            std::cout << "Skipping gyro data. Timestamp before the first accel "
                         "measurement.";
            gyro_data_queue.pop_front();
          }

          while (!gyro_data_queue.empty() &&
                 gyro_data_queue.front().timestamp < d.timestamp) {
            RsIMUData gyro_data = gyro_data_queue.front();
            gyro_data_queue.pop_front();

            double w0 = (d.timestamp - gyro_data.timestamp) /
                        (d.timestamp - prev_accel_data->timestamp);

            double w1 = (gyro_data.timestamp - prev_accel_data->timestamp) /
                        (d.timestamp - prev_accel_data->timestamp);

            Eigen::Vector3d accel_interpolated =
                w0 * prev_accel_data->data + w1 * d.data;

            basalt::ImuData<double>::Ptr data;
            data.reset(new basalt::ImuData<double>);
            data->t_ns = gyro_data.timestamp * 1e6;
            data->accel = accel_interpolated;
            data->gyro = gyro_data.data;

            if (imu_data_queue) imu_data_queue->push(data);
          }

          prev_accel_data.reset(new RsIMUData(d));
        }
      }
    } else if (auto fs = frame.as<rs2::frameset>()) { // always get error after first frame obtained
      // (sensor.cpp:572) acquire_power failed: map_device_descriptor Cannot open '/dev/video0 Last Error: No such device
      // (global_timestamp_reader.cpp:239) Error during time_diff_keeper polling: map_device_descriptor Cannot open '/dev/video0 Last Error: No such device
      double new_timestamp_image = fs.get_timestamp()*1e-3;
      if(abs(timestamp_image-new_timestamp_image)<0.001){
        frame_counter--;
        return;
      }
//      if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
//      {
//        //If the profile was changed, update the align object, and also get the new device's depth scale
//        profile = pipe.get_active_profile();
//        align_to = find_stream_to_align(profile.get_streams());
//        align = new rs2::align(align_to);
//      }
      std::cout << "frame set size: " << fs.size() << std::endl;
      std::vector<rs2::video_frame> vfs;
      for (int i = 0; i < NUM_CAMS; i++) {
        rs2::video_frame vf = fs[i].as<rs2::video_frame>();
        if (!vf) {
          std::cout << "Weird Frame, skipping" << std::endl;
          return;
        }
        vfs.push_back(vf);
      }

      // skip frames if configured
      if (frame_counter++ % skip_frames != 0) {
        return;
      }
      std::cout << "frame count: " << frame_counter << std::endl;
      OpticalFlowInput::Ptr data(new OpticalFlowInput);
      data->img_data.resize(NUM_CAMS);

      for (int i = 0; i < NUM_CAMS; i++) {
        const auto& vf = vfs[i];

        int64_t t_ns = vf.get_timestamp() * 1e6;

        data->t_ns = t_ns;

        data->img_data[i].exposure =
            vf.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE) * 1e-6;

        data->img_data[i].img.reset(new basalt::ManagedImage<uint16_t>(
            vf.get_width(), vf.get_height()));
        const uint16_t* data_in = (const uint16_t*)vf.get_data();
        uint16_t* data_out = data->img_data[i].img->ptr;
        size_t full_size = vf.get_width() * vf.get_height();
        for (size_t j = 0; j < full_size; j++) {
          data_out[j] = data_in[j];
        }
        /*
        data->img_data[i].img.reset(new basalt::ManagedImage<uint16_t>(
            vf.get_width(), vf.get_height()));
        const uint8_t* data_in = (const uint8_t*)vf.get_data();
        uint16_t* data_out = data->img_data[i].img->ptr;

        size_t full_size = vf.get_width() * vf.get_height();
        for (size_t j = 0; j < full_size; j++) {
          int val = data_in[j];
          val = val << 8;
          data_out[j] = val;
        } */

        std::cout << "Timestamp / exposure " << i << ": " << data->t_ns << " / "
                  << int(data->img_data[i].exposure * 1e3) << "ms" << std::endl;
      }

      last_img_data = data;
      if (image_data_queue) image_data_queue->push(data);
      std::cout << "done img data reading ..." << std::endl;
    } else if (auto pf = frame.as<rs2::pose_frame>()) {
      auto data = pf.get_pose_data();

      RsPoseData pdata;
      pdata.t_ns = pf.get_timestamp() * 1e6;

      Eigen::Vector3d trans(data.translation.x, data.translation.y,
                            data.translation.z);
      Eigen::Quaterniond quat(data.rotation.w, data.rotation.x, data.rotation.y,
                              data.rotation.z);

      pdata.data = Sophus::SE3d(quat, trans);

      if (pose_data_queue) pose_data_queue->push(pdata);
    }
  };

  profile = pipe.start(config, callback);
}

void RsL515Device::stop() {
  if (image_data_queue) image_data_queue->push(nullptr);
  if (imu_data_queue) imu_data_queue->push(nullptr);
}

bool RsL515Device::setExposure(double exposure) {
  if (!manual_exposure) return false;

  sensor.set_option(rs2_option::RS2_OPTION_EXPOSURE, exposure * 1000);
  return true;
}

bool RsL515Device::profile_changed(
    const std::vector<rs2::stream_profile>& current,
    const std::vector<rs2::stream_profile>& prev) {
  for (auto&& sp : prev) {
    // If previous profile is in current (maybe just added another)
    auto itr = std::find_if(std::begin(current), std::end(current),
                            [&sp](const rs2::stream_profile& current_sp) {
                              return sp.unique_id() == current_sp.unique_id();
                            });
    if (itr ==
        std::end(current))  // If it previous stream wasn't found in current
    {
      return true;
    }
  }
  return false;
}

rs2_stream RsL515Device::find_stream_to_align(
    const std::vector<rs2::stream_profile>& streams) {
  // Given a vector of streams, we try to find a depth stream and another stream
  // to align depth with. We prioritize color streams to make the view look
  // better. If color is not available, we take another stream that (other than
  // depth)
  align_to = RS2_STREAM_ANY;
  bool depth_stream_found = false;
  bool color_stream_found = false;
  for (rs2::stream_profile sp : streams) {
    rs2_stream profile_stream = sp.stream_type();
    if (profile_stream != RS2_STREAM_DEPTH) {
      if (!color_stream_found)  // Prefer color
        align_to = profile_stream;

      if (profile_stream == RS2_STREAM_COLOR) {
        color_stream_found = true;
      }
    } else {
      depth_stream_found = true;
    }
  }

  if (!depth_stream_found)
    throw std::runtime_error("No Depth stream available");

  if (align_to == RS2_STREAM_ANY)
    throw std::runtime_error("No stream found to align with Depth");

  return align_to;
}

void RsL515Device::setSkipFrames(int skip) { skip_frames = skip; }

void RsL515Device::setWebpQuality(int quality) { webp_quality = quality; }

std::shared_ptr<basalt::Calibration<double>> RsL515Device::exportCalibration() {
  using Scalar = double;

  if (calib.get()) return calib;

  calib.reset(new basalt::Calibration<Scalar>);
  calib->imu_update_rate = IMU_RATE;

  auto accel_stream = profile.get_stream(RS2_STREAM_ACCEL);
  auto gyro_stream = profile.get_stream(RS2_STREAM_GYRO);
  auto cam0_stream = profile.get_stream(RS2_STREAM_COLOR);

  // get gyro extrinsics
  if (auto gyro = gyro_stream.as<rs2::motion_stream_profile>()) {
    rs2_motion_device_intrinsic intrinsics = gyro.get_motion_intrinsics();

    Eigen::Matrix<Scalar, 12, 1> gyro_bias_full;
    gyro_bias_full << intrinsics.data[0][3], intrinsics.data[1][3],
        intrinsics.data[2][3], intrinsics.data[0][0] - 1.0,
        intrinsics.data[1][0], intrinsics.data[2][0], intrinsics.data[0][1],
        intrinsics.data[1][1] - 1.0, intrinsics.data[2][1],
        intrinsics.data[0][2], intrinsics.data[1][2],
        intrinsics.data[2][2] - 1.0;
    basalt::CalibGyroBias<Scalar> gyro_bias;
    gyro_bias.getParam() = gyro_bias_full;
    calib->calib_gyro_bias = gyro_bias;

    // std::cout << "Gyro Bias\n" << gyro_bias_full << std::endl;

    calib->gyro_noise_std = Eigen::Vector3d(intrinsics.noise_variances[0],
                                            intrinsics.noise_variances[1],
                                            intrinsics.noise_variances[2])
                                .cwiseSqrt();

    calib->gyro_bias_std = Eigen::Vector3d(intrinsics.bias_variances[0],
                                           intrinsics.bias_variances[1],
                                           intrinsics.bias_variances[2])
                               .cwiseSqrt();

    // std::cout << "Gyro noise var: " << intrinsics.noise_variances[0]
    //          << " bias var: " << intrinsics.bias_variances[0] << std::endl;
  } else {
    std::abort();
  }

  // get accel extrinsics
  if (auto accel = accel_stream.as<rs2::motion_stream_profile>()) {
    rs2_motion_device_intrinsic intrinsics = accel.get_motion_intrinsics();
    Eigen::Matrix<Scalar, 9, 1> accel_bias_full;
    accel_bias_full << intrinsics.data[0][3], intrinsics.data[1][3],
        intrinsics.data[2][3], intrinsics.data[0][0] - 1.0,
        intrinsics.data[1][0], intrinsics.data[2][0],
        intrinsics.data[1][1] - 1.0, intrinsics.data[2][1],
        intrinsics.data[2][2] - 1.0;
    basalt::CalibAccelBias<Scalar> accel_bias;
    accel_bias.getParam() = accel_bias_full;
    calib->calib_accel_bias = accel_bias;

    // std::cout << "Gyro Bias\n" << accel_bias_full << std::endl;

    calib->accel_noise_std = Eigen::Vector3d(intrinsics.noise_variances[0],
                                             intrinsics.noise_variances[1],
                                             intrinsics.noise_variances[2])
                                 .cwiseSqrt();

    calib->accel_bias_std = Eigen::Vector3d(intrinsics.bias_variances[0],
                                            intrinsics.bias_variances[1],
                                            intrinsics.bias_variances[2])
                                .cwiseSqrt();

    // std::cout << "Accel noise var: " << intrinsics.noise_variances[0]
    //          << " bias var: " << intrinsics.bias_variances[0] << std::endl;
  } else {
    std::abort();
  }

  // get camera ex-/intrinsics
  for (const auto& cam_stream : {cam0_stream}) {
    if (auto cam = cam_stream.as<rs2::video_stream_profile>()) {
      // extrinsics
      rs2_extrinsics ex = cam.get_extrinsics_to(gyro_stream);
      Eigen::Matrix3f rot = Eigen::Map<Eigen::Matrix3f>(ex.rotation);
      Eigen::Vector3f trans = Eigen::Map<Eigen::Vector3f>(ex.translation);

      Eigen::Quaterniond q(rot.cast<double>());
      basalt::Calibration<Scalar>::SE3 T_i_c(q, trans.cast<double>());

      // std::cout << "T_i_c\n" << T_i_c.matrix() << std::endl;

      calib->T_i_c.push_back(T_i_c);

      // get resolution
      Eigen::Vector2i resolution;
      resolution << cam.width(), cam.height();
      calib->resolution.push_back(resolution);

      // intrinsics
      rs2_intrinsics intrinsics = cam.get_intrinsics();
      basalt::KannalaBrandtCamera4<Scalar>::VecN params;
      params << intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
          intrinsics.coeffs[0], intrinsics.coeffs[1], intrinsics.coeffs[2],
          intrinsics.coeffs[3];

      // std::cout << "Camera intrinsics: " << params.transpose() << std::endl;

      basalt::GenericCamera<Scalar> camera;
      basalt::KannalaBrandtCamera4 kannala_brandt(params);
      camera.variant = kannala_brandt;

      calib->intrinsics.push_back(camera);
    } else {
      std::abort();
    }
  }
  std::cout << "got calib" << std::endl;
  return calib;
}

}  // namespace basalt
