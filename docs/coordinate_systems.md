# AirMSPI 仿真坐标系与转换关系（图文版）

> 版本说明：配合 `scripts/airmspi_image_simulation.py` 当前实现。

## 1) 坐标系定义

### A. 世界坐标系（World）
- `x`: North (+)
- `y`: East (+)
- `z`: Up (+)

> 结论：这是 **NEU** 轴顺序（`x=N, y=E, z=U`）。按右手定则检验：`x × y = N × E = -U`，因此该轴顺序在几何上是**左手系表示**。
>
> 若你外部数据按常见 **ENU 右手系**（`x=E, y=N, z=U`），则需要交换 `x/y`（以及所有以 `+x` 为零方位参考的角度解释）。

用于：
- 传感器位置 `position_vector`
- 视线方向 `lookat_vector`
- 角度产品（`VZA/VAA/RAA/Scattering Angle`）

### B. 相机像平面坐标（Camera Pixel）
- 由 `x_resolution/y_resolution` 与相机模型生成。
- 这是“飞机上看到的二维图像阵列”。

### C. 地面投影坐标（Ground）
- 由射线与地面 `z=0` 相交得到 `x_ground/y_ground`。
- 用于 `registered` 与 `downsampled_registered`。

### D. 地理坐标（Geodetic）
- `lat/lon` 来自 txt 或 fallback 规则。
- 用于元数据、定位与重绘参考，不直接参与辐射传输求解。

---

## 2) 转换流程图（Mermaid）

```mermaid
flowchart LR
  A[TXT cloud fields\n(x,y,z,cv,reff,veff,lat,lon)] --> B[World cloud grid]
  B --> C[AT3D RTE solver\n(I,Q,U in sensor image)]
  C --> D[Camera Pixel Array]
  D --> E[Orientation transform\ntranspose/flip (config)]
  E --> F[Compute v_out_map]\nVZA/VAA/RAA/SCA
  E --> G[Reproject to ground\n(x_ground,y_ground)]
  G --> H[registered]
  H --> I[downsampled_registered]
  B --> J[lat/lon reference grid]
  G --> K[assign_latlon_from_grid]
  J --> K
  K --> H
```

> 在支持 Mermaid 的 Markdown 查看器中可交互缩放/高亮节点。

---

## 3) 与“方向”相关的配置（核心）

位于 `sensor.trajectory`：

- `trajectory.mode`
  - `adjacent_views`: 用相邻视角位置差分推断 heading。
  - `sensor_azimuth`: 直接用 `views.azimuth_deg` 作为 heading。
- `manual_flight_azimuth_deg`
  - 旧字段兼容保留，不再推荐作为主 heading 控制项。
- `camera_align_with_flight_heading`
  - `false`（默认）：相机上方向与世界上方向对齐，避免隐式随航向旋转。
  - `true`：相机上方向跟随飞机航向。
- `camera_relative_roll_deg`
  - 相机相对飞机滚转角（绕视线方向）。
- `camera_image_transpose` / `camera_image_flip_lr`
  - 仅用于“像元排布兼容”开关（历史观测方向对齐），不建议默认开启。

---

## 4) 你当前推荐设置（对应“所见即所得”）

```yaml
sensor:
  trajectory:
    mode: "sensor_azimuth"
    camera_align_with_flight_heading: false
    camera_relative_roll_deg: 0.0
    camera_image_transpose: false
    camera_image_flip_lr: false
```

这组设置的含义：
- VAA 角度仍按世界坐标定义；
- 不额外引入相机-飞机相对滚转；
- 不做历史兼容的转置/翻转；
- 原始模拟图像方向更加“物理直观”。

---

## 5) 常见“看起来被转置/翻转”的来源

1. 观测产品像元排布与模型像平面定义不同（历史兼容常需转置/翻转）。
2. 相机上方向跟随飞机航向，导致图像坐标轴相对世界旋转。
3. 绘图时从 Camera Pixel 切换到 Ground（registered）后，坐标语义改变。

---

## 6) Aircraft 姿态模式下的角度约定（排错重点）

- 在当前 NEU 定义下，`+x=North, +y=East, +z=Up`。
- 若 `heading` 定义为“**从 North 开始顺时针**”（航空常用）：
  - `heading=0°` 指向 `+x`（North）
  - `heading=90°` 指向 `+y`（East）
- 代码实现应满足：`math_rotation = -heading`（绕 `+z` 轴）。

若这里误用了“从 +y 起算”或多做了 `90°` 偏移，通常会出现“地面 registered 整体歪斜/旋转”的症状。
