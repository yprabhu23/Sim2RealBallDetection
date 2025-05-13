import pyrealsense2 as rs

# Create a context object. This manages the connection to the device
pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream (you can also enable depth, etc.)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline_profile = pipeline.start(config)

# Get stream profile and extract camera intrinsics
color_stream = pipeline_profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

print("Width:", intrinsics.width)
print("Height:", intrinsics.height)
print("Focal Length (fx, fy):", intrinsics.fx, intrinsics.fy)
print("Principal Point (ppx, ppy):", intrinsics.ppx, intrinsics.ppy)
print("Distortion Model:", intrinsics.model)
print("Distortion Coefficients:", intrinsics.coeffs)

# Stop the pipeline after getting data
pipeline.stop()