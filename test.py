def get_n_joint_xml(num_joints, link_lengths=None, with_obstacles=False):
    """通用N关节XML配置生成器（支持1-10关节）
    
    Args:
        num_joints: 关节数量（1-10）
        link_lengths: 各关节长度列表
        with_obstacles: 是否包含障碍物
    """
    # if link_lengths is None:
    #     # 默认长度：根据关节数自适应
    #     default_length = max(0.04, 0.3 / num_joints)
    #     link_lengths = [default_length] * num_joints
    

    if link_lengths is None:
        raise ValueError(f"get_n_joint_xml: link_lengths 不能为空")

    if len(link_lengths) != num_joints:
        raise ValueError(f"get_n_joint_xml: link_lengths 长度（{len(link_lengths)}）必须等于关节数量（{num_joints}）")    


    # 生成链接的XML字符串
    links_xml = []
    current_pos = 0.0
    
    for i in range(num_joints):
        length = link_lengths[i]
        
        # 第一个关节
        if i == 0:
            link_xml = f"""
    <body name="body{i}" pos="0 0 .01">
      <geom fromto="0 0 0 {length} 0 0" name="link{i}" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="{1+i}" conaffinity="255"/>
      <joint axis="0 0 1" limited="false" name="joint{i}" pos="0 0 0" type="hinge"/>"""
        else:
            # 后续关节
            link_xml = f"""
      <body name="body{i}" pos="{link_lengths[i-1]} 0 0">
        <joint axis="0 0 1" limited="true" name="joint{i}" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 {length} 0 0" name="link{i}" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="{1+i}" conaffinity="255"/>"""
        
        links_xml.append(link_xml)
        current_pos += length
    
    # 末端 - fingertip位置：直接放在link末端，避免"悬空"
    # fingertip球体半径0.01m，中心放在link末端位置，球体会自然"包裹"末端
    fingertip_pos = link_lengths[-1]
    fingertip_xml = f"""
        <body name="fingertip" pos="{fingertip_pos} 0 0">
          <geom contype="{1+num_joints}" conaffinity="255" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
        </body>"""
    
    # 关闭所有body标签
    close_tags = "      </body>\n" * (num_joints - 1) + "    </body>"
    
    # 生成自碰撞对（对于6+关节）
    collision_pairs = ""
    if num_joints >= 6:
        pairs = []
        for i in range(num_joints):
            for j in range(i+2, num_joints):
                pairs.append(f'    <pair geom1="link{i}" geom2="link{j}" condim="3"/>')
        collision_pairs = "\n".join(pairs)
    
    # 🎯 根据关节长度动态计算边框大小
    max_reach = sum(link_lengths)  # 最大可达半径
    # 边框大小 = 最大可达范围 * 1.5，确保Reacher不会穿出边框
    border_size = max(0.3, max_reach * 1.5)  # 至少保持0.3的最小边框
    half_border = border_size
    
    # 目标范围：略小于边框，避免目标生成在边界上
    target_range = border_size * 0.9
    
    # 障碍物XML
    obstacles_xml = ""
    if with_obstacles:
        # 如果with_obstacles是字典，支持自定义障碍物配置
        if isinstance(with_obstacles, dict):
            obs_type = with_obstacles.get('type', 'gap')
            gap_width = with_obstacles.get('gap_width', 0.10)
            wall_distance = with_obstacles.get('wall_distance', 0.15)
            
            if obs_type == 'gap':
                # 🔄 重新设计：缝隙型障碍物放在Y轴下方（而不是X轴右边）
                # 布局：Reacher(0,0) 向下看 → 水平墙(Y=-wall_distance) 中间有缝隙(X=0) → 目标在墙下方
                wall_length = 0.15
                wall_thickness = 0.08  # 🔧 大幅增加墙厚度：0.02 → 0.08 (8cm)，防止穿透
                gap_half = gap_width / 2.0
                
                # 🔧 关键：墙的上边缘（Reacher侧）应该在Y=-wall_distance
                # box的size是half-extents，所以墙的体积是：
                #   Y方向: [center_y - thickness/2, center_y + thickness/2]
                # 我们要上边缘在-wall_distance，所以：
                #   center_y - thickness/2 = -wall_distance
                #   center_y = -wall_distance + thickness/2
                # 但为了视觉上清晰，墙的中心就放在-wall_distance，向下延伸
                wall_center_y = wall_distance + wall_thickness / 2.0
                
                obstacles_xml = f"""
    <!-- 缝隙型障碍物：水平布局（在下方Y轴负方向）-->
    <body name="wall_left" pos="-{gap_half + wall_thickness/2} -{wall_center_y} .01">
      <geom name="wall_left" type="box" size="{wall_length/2} {wall_thickness/2} 0.05" rgba="0.3 0.3 0.3 1.0" contype="32" conaffinity="255"/>
    </body>
    <body name="wall_right" pos="{gap_half + wall_thickness/2} -{wall_center_y} .01">
      <geom name="wall_right" type="box" size="{wall_length/2} {wall_thickness/2} 0.05" rgba="0.3 0.3 0.3 1.0" contype="32" conaffinity="255"/>
    </body>"""
            elif obs_type == 'u_shape':
                # U型通道
                obstacles_xml = f"""
    <!-- U型通道障碍物 -->
    <body name="wall_left" pos="{wall_distance - 0.05} -0.10 .01">
      <geom name="wall_left" type="box" size="0.01 0.10 0.02" rgba="0.3 0.3 0.3 1.0" contype="16" conaffinity="31"/>
    </body>
    <body name="wall_right" pos="{wall_distance - 0.05} 0.10 .01">
      <geom name="wall_right" type="box" size="0.01 0.10 0.02" rgba="0.3 0.3 0.3 1.0" contype="16" conaffinity="31"/>
    </body>
    <body name="wall_back" pos="{wall_distance + 0.05} 0 .01">
      <geom name="wall_back" type="box" size="0.10 0.01 0.02" rgba="0.3 0.3 0.3 1.0" contype="16" conaffinity="31"/>
    </body>"""
        else:
            # 默认：简单的球形障碍物
            obstacles_xml = """
    <body name="obstacle1" pos="0.15 0.05 .01">
      <geom name="obstacle1" type="sphere" size=".03" rgba="0.9 0.1 0.1 0.6" contype="16" conaffinity="16"/>
    </body>
    <body name="obstacle2" pos="0.10 -0.08 .01">
      <geom name="obstacle2" type="sphere" size=".03" rgba="0.9 0.1 0.1 0.6" contype="16" conaffinity="16"/>
    </body>"""
    
    # 组装完整XML（🎨 动态边框：根据关节数量自适应）
    xml = f"""
<mujoco model="{num_joints}joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
  </default>
  {"<contact>" + collision_pairs + "</contact>" if collision_pairs else ""}
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="{border_size} {border_size} 10" type="plane"/>
    <geom conaffinity="0" contype="0" fromto="-{half_border} -{half_border} .01 {half_border} -{half_border} .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto=" {half_border} -{half_border} .01 {half_border}  {half_border} .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-{half_border}  {half_border} .01 {half_border}  {half_border} .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-{half_border} -{half_border} .01 -{half_border} {half_border} .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    {"".join(links_xml)}
{fingertip_xml}
{close_tags}
{obstacles_xml}
    <body name="target" pos=".1 -.1 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-{target_range} {target_range}" ref=".1" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-{target_range} {target_range}" ref="-.1" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    {"".join([f'<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint{i}"/>' for i in range(num_joints)])}
  </actuator>
</mujoco>
"""
    return xml


xml = get_n_joint_xml(
    num_joints=3,
    link_lengths=[0.10, 0.10, 0.10],   # ✅ 长度必须=3
    with_obstacles=False               # ✅ 可选
)
print(xml)  # 先看前 500 字符确认有输出