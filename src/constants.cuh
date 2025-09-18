#pragma once

constexpr float ASPECT_RATIO      = 1.7777777; // 16:9
constexpr int   IMAGE_WIDTH       = 1280;
constexpr int   IMAGE_HEIGHT      = (int)(IMAGE_WIDTH / ASPECT_RATIO);
constexpr int   BYTES_PER_PIXEL   = 3;
constexpr int   SAMPLES_PER_PIXEL = 250;
constexpr int   MAX_BOUNCES       = 50;