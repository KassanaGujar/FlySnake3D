extends Node3D
## FlySnake3D replay viewer.
##
## Python is authoritative: this scene only draws frames recorded by
## game/runner.py. It never simulates or changes the game.
##
## Command line (everything after "--" goes to the viewer):
##   --replay=<path.jsonl>   open a replay
##   --check                 headless: load + validate, print VIEWER_CHECK {json}, quit
##   --screenshot=<png>      render one frame to a PNG and quit
##   --frame=<n|trap|last>   frame for --screenshot (default last)
##   --cam=<1-4>             camera for --screenshot
##
## Keys: Space play/pause, Left/Right step, Up/Down speed, Home/End,
##       T jump to first trap, 1-4 cameras, O open file, H toggle help.

const ReplayLoader = preload("res://replay_loader.gd")

enum Cam { ORBIT, TOP, CHASE, HEAD }
const CAM_NAMES := ["orbit (drag / wheel)", "top-down", "chase", "head position"]

const COL_HEAD := Color(0.35, 1.0, 0.45)
const COL_NECK := Color(0.2, 0.75, 0.35)
const COL_TAIL := Color(0.05, 0.2, 0.1)
const COL_TAIL_TIP := Color(1.0, 0.7, 0.2)
const COL_FOOD := Color(1.0, 0.25, 0.2)
const COL_PLAN := Color(0.3, 0.85, 1.0, 0.7)
const COL_TRAP := Color(1.0, 0.35, 0.3)

var header: Dictionary = {}
var frames: Array = []
var replay_path := ""
var idx := 0
var playing := false
var fps := 10.0
var acc := 0.0
var cam_mode: int = Cam.ORBIT
var W := 20
var H := 20
var first_trap := -1
var show_help := true

var orbit_yaw := 0.7
var orbit_pitch := 0.95
var orbit_dist := 32.0
var dragging := false

var camera: Camera3D
var arena: Node3D
var body_mm: MultiMeshInstance3D
var plan_mm: MultiMeshInstance3D
var head_node: MeshInstance3D
var food_node: MeshInstance3D
var info: Label
var status: Label
var help: Label
var slider: HSlider
var dialog: FileDialog


func _ready() -> void:
	var args := _args()
	if args.has("brain") and not Engine.has_meta("left_brain"):
		get_tree().change_scene_to_file.call_deferred("res://brain.tscn")
		return
	_build_world()
	_build_hud()
	var path: String = args.get("replay", "")
	var root := ProjectSettings.globalize_path("res://").path_join("..").simplify_path()
	if path != "" and path.is_relative_path():
		path = root.path_join(path).simplify_path()
	if path == "" and Engine.has_meta("replay_path"):
		path = Engine.get_meta("replay_path")
	if path == "" and not args.has("check"):
		# open the first greedy A* self-trap replay by default
		var d := root.path_join("results/replays/stage2/astar_greedy_seed0.jsonl")
		if FileAccess.file_exists(d):
			path = d
	if path != "":
		_open(path)
	if args.has("check"):
		_headless_check()
		return
	if args.has("screenshot"):
		cam_mode = clampi(int(args.get("cam", "1")) - 1, 0, 3)
		var fr: String = args.get("frame", "last")
		if fr == "trap" and first_trap >= 0:
			_show(first_trap)
		elif fr == "last" or fr == "trap":
			_show(frames.size() - 1)
		else:
			_show(int(fr))
		show_help = false
		help.visible = false
		_take_screenshot(args["screenshot"])
		return
	if path == "":
		status.text = "Press O to open a replay (results/replays/), B for the brain view."


func _args() -> Dictionary:
	var d := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv := a.substr(2).split("=", true, 1)
			d[kv[0]] = kv[1] if kv.size() > 1 else "true"
	return d


# ------------------------------------------------------------------ loading
func _open(path: String) -> void:
	var r := ReplayLoader.load_replay(path)
	if r.has("error"):
		push_error(r["error"])
		if status:
			status.text = "Error: " + r["error"]
		return
	header = r["header"]
	frames = r["frames"]
	replay_path = path
	Engine.set_meta("replay_path", path)
	W = int(header["config"]["width"])
	H = int(header["config"]["height"])
	var ft = header["result"].get("first_trap_step")
	first_trap = int(ft) if ft != null else -1
	orbit_dist = float(maxi(W, H)) * 1.9
	_build_arena()
	slider.max_value = frames.size() - 1
	playing = false
	_show(0)


func _headless_check() -> void:
	var out := {"ok": false, "frames": frames.size()}
	if frames.is_empty():
		out["problems"] = ["no replay loaded"]
	else:
		var problems := ReplayLoader.validate(header, frames)
		# Exercise the drawing path on every frame as well.
		for i in frames.size():
			_show(i)
		out["problems"] = problems
		out["ok"] = problems.is_empty()
		out["agent"] = header["result"]["agent"]
		out["drawn_segments_last"] = body_mm.multimesh.visible_instance_count + 1
	print("VIEWER_CHECK " + JSON.stringify(out))
	get_tree().quit(0 if out["ok"] else 1)


func _take_screenshot(path: String) -> void:
	_update_camera(1.0)
	for i in 4:
		await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	var err := img.save_png(path)
	print("VIEWER_SCREENSHOT %s %s" % [path, error_string(err)])
	get_tree().quit(0 if err == OK else 1)


# ------------------------------------------------------------------ world
func _mat(c: Color, unshaded := false, emit := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	if unshaded:
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	if c.a < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	if emit > 0.0:
		m.emission_enabled = true
		m.emission = c
		m.emission_energy_multiplier = emit
	return m


func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.05, 0.055, 0.07)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.6, 0.65, 0.75)
	env.ambient_light_energy = 0.45
	env.glow_enabled = true
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, 35, 0)
	sun.light_energy = 1.1
	sun.shadow_enabled = true
	add_child(sun)

	camera = Camera3D.new()
	camera.current = true
	add_child(camera)

	# Snake body: one multimesh instance per segment (head drawn separately).
	var seg := BoxMesh.new()
	seg.size = Vector3(0.84, 0.6, 0.84)
	var seg_mat := StandardMaterial3D.new()
	seg_mat.vertex_color_use_as_albedo = true
	seg.material = seg_mat
	body_mm = MultiMeshInstance3D.new()
	body_mm.multimesh = MultiMesh.new()
	body_mm.multimesh.transform_format = MultiMesh.TRANSFORM_3D
	body_mm.multimesh.use_colors = true
	body_mm.multimesh.mesh = seg
	add_child(body_mm)

	var dot := BoxMesh.new()
	dot.size = Vector3(0.28, 0.04, 0.28)
	dot.material = _mat(COL_PLAN, true)
	plan_mm = MultiMeshInstance3D.new()
	plan_mm.multimesh = MultiMesh.new()
	plan_mm.multimesh.transform_format = MultiMesh.TRANSFORM_3D
	plan_mm.multimesh.mesh = dot
	add_child(plan_mm)

	# Head with a "nose" on local -Z so the heading is visible.
	head_node = MeshInstance3D.new()
	var hb := BoxMesh.new()
	hb.size = Vector3(0.94, 0.72, 0.94)
	head_node.mesh = hb
	head_node.material_override = _mat(COL_HEAD, false, 0.4)
	var nose := MeshInstance3D.new()
	var nb := BoxMesh.new()
	nb.size = Vector3(0.4, 0.3, 0.3)
	nose.mesh = nb
	nose.position = Vector3(0, 0.1, -0.55)
	nose.material_override = _mat(Color(0.95, 1.0, 0.6), false, 0.6)
	head_node.add_child(nose)
	add_child(head_node)

	food_node = MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.35
	sm.height = 0.7
	food_node.mesh = sm
	food_node.material_override = _mat(COL_FOOD, false, 1.5)
	add_child(food_node)


func _build_arena() -> void:
	if arena:
		arena.queue_free()
	arena = Node3D.new()
	add_child(arena)
	var center := _center()

	var floor_mi := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(W, H)
	floor_mi.mesh = pm
	floor_mi.position = Vector3(center.x, 0, center.z)
	floor_mi.material_override = _mat(Color(0.13, 0.14, 0.17))
	arena.add_child(floor_mi)

	var im := ImmediateMesh.new()
	im.surface_begin(Mesh.PRIMITIVE_LINES, _mat(Color(0.24, 0.26, 0.31), true))
	for x in range(W + 1):
		im.surface_add_vertex(Vector3(x - 0.5, 0.01, -0.5))
		im.surface_add_vertex(Vector3(x - 0.5, 0.01, H - 0.5))
	for z in range(H + 1):
		im.surface_add_vertex(Vector3(-0.5, 0.01, z - 0.5))
		im.surface_add_vertex(Vector3(W - 0.5, 0.01, z - 0.5))
	im.surface_end()
	var grid := MeshInstance3D.new()
	grid.mesh = im
	arena.add_child(grid)

	var wall_mat := _mat(Color(0.32, 0.36, 0.45))
	var t := 0.4
	for spec in [
		[Vector3(center.x, 0.3, -0.5 - t / 2), Vector3(W + 2 * t, 0.6, t)],
		[Vector3(center.x, 0.3, H - 0.5 + t / 2), Vector3(W + 2 * t, 0.6, t)],
		[Vector3(-0.5 - t / 2, 0.3, center.z), Vector3(t, 0.6, H)],
		[Vector3(W - 0.5 + t / 2, 0.3, center.z), Vector3(t, 0.6, H)],
	]:
		var wm := MeshInstance3D.new()
		var b := BoxMesh.new()
		b.size = spec[1]
		wm.mesh = b
		wm.position = spec[0]
		wm.material_override = wall_mat
		arena.add_child(wm)

	for o in header["config"].get("obstacles", []):
		var om := MeshInstance3D.new()
		var ob := BoxMesh.new()
		ob.size = Vector3(0.95, 0.9, 0.95)
		om.mesh = ob
		om.position = _p(o, 0.45)
		om.material_override = wall_mat
		arena.add_child(om)

	body_mm.multimesh.instance_count = W * H
	body_mm.multimesh.visible_instance_count = 0
	plan_mm.multimesh.instance_count = W * H
	plan_mm.multimesh.visible_instance_count = 0


func _center() -> Vector3:
	return Vector3(W / 2.0 - 0.5, 0, H / 2.0 - 0.5)


## Grid cell (x, y) -> world position. Grid y maps to world z.
func _p(c, y: float) -> Vector3:
	return Vector3(float(c[0]), y, float(c[1]))


# ------------------------------------------------------------------ drawing
func _show(i: int) -> void:
	if frames.is_empty():
		return
	idx = clampi(i, 0, frames.size() - 1)
	var fr: Dictionary = frames[idx]
	var body: Array = fr["body"]
	var n := body.size()

	var mm := body_mm.multimesh
	mm.visible_instance_count = mini(maxi(n - 1, 0), mm.instance_count)
	for k in range(1, mini(n, mm.instance_count + 1)):
		mm.set_instance_transform(k - 1, Transform3D(Basis(), _p(body[k], 0.3)))
		var tt: float = float(k - 1) / float(maxi(n - 2, 1))
		mm.set_instance_color(k - 1, COL_TAIL_TIP if k == n - 1 else COL_NECK.lerp(COL_TAIL, tt))

	head_node.position = _p(body[0], 0.36)
	var hd := _heading()
	head_node.basis = Basis.looking_at(hd, Vector3.UP)

	food_node.visible = fr.get("food") != null
	if food_node.visible:
		food_node.position = _p(fr["food"], 0.4)

	var plan: Array = fr.get("plan", [])
	var pmm := plan_mm.multimesh
	pmm.visible_instance_count = mini(plan.size(), pmm.instance_count)
	for k in pmm.visible_instance_count:
		pmm.set_instance_transform(k, Transform3D(Basis(), _p(plan[k], 0.03)))

	_update_hud()
	if slider:
		slider.set_value_no_signal(idx)


func _heading() -> Vector3:
	if frames.is_empty():
		return Vector3.FORWARD
	var h: Array = frames[idx]["heading"]
	return Vector3(float(h[0]), 0, float(h[1]))


# ------------------------------------------------------------------ camera
func _update_camera(delta: float) -> void:
	var center := _center()
	var w := clampf(delta * 6.0, 0.0, 1.0)
	match cam_mode:
		Cam.ORBIT:
			camera.projection = Camera3D.PROJECTION_PERSPECTIVE
			camera.fov = 50
			var off := Vector3(
				cos(orbit_pitch) * sin(orbit_yaw),
				sin(orbit_pitch),
				cos(orbit_pitch) * cos(orbit_yaw)) * orbit_dist
			camera.position = center + off
			camera.look_at(center, Vector3.UP)
		Cam.TOP:
			camera.projection = Camera3D.PROJECTION_ORTHOGONAL
			camera.size = float(maxi(W, H)) + 2.0
			camera.position = center + Vector3(0, 60, 0)
			camera.look_at(center, Vector3(0, 0, -1))
		Cam.CHASE, Cam.HEAD:
			if frames.is_empty():
				return
			camera.projection = Camera3D.PROJECTION_PERSPECTIVE
			var head := head_node.position
			var hd := _heading()
			var want_pos: Vector3
			var want_look: Vector3
			if cam_mode == Cam.CHASE:
				camera.fov = 60
				want_pos = head - hd * 7.0 + Vector3.UP * 6.0
				want_look = head + hd * 3.0
			else:
				# Approximate eye position only; the fly's visual input is
				# built by the encoder in Python, not by this camera.
				camera.fov = 100
				want_pos = head + Vector3.UP * 0.55
				want_look = want_pos + hd * 5.0
			camera.position = camera.position.lerp(want_pos, w) if w < 1.0 else want_pos
			var cur_look := camera.position - camera.global_basis.z
			var look: Vector3 = cur_look.lerp(want_look, w) if w < 1.0 else want_look
			if not camera.position.is_equal_approx(look):
				camera.look_at(look, Vector3.UP)


# ------------------------------------------------------------------ HUD
func _label(pos: Vector2, size: int) -> Label:
	var l := Label.new()
	l.position = pos
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_constant_override("outline_size", 5)
	l.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.9))
	return l


func _build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	info = _label(Vector2(14, 10), 17)
	layer.add_child(info)
	status = _label(Vector2(14, 0), 22)
	status.set_anchors_preset(Control.PRESET_BOTTOM_LEFT)
	status.position = Vector2(14, -78)
	layer.add_child(status)
	help = _label(Vector2(0, 10), 14)
	help.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	help.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	help.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	help.position = Vector2(-14, 10)
	help.text = "\n".join([
		"Space  play / pause", "Left/Right  step", "Up/Down  speed",
		"Home/End  start / end", "T  first trap", "1-4  camera", "O  open replay",
		"B  fly brain view",
		"H  hide help", "",
		"green box = head (nose = heading)", "body darkens towards the tail", "amber = tail tip",
		"red = food", "cyan dots = planner's path (A* only)"])
	layer.add_child(help)

	slider = HSlider.new()
	slider.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	slider.offset_left = 14
	slider.offset_right = -14
	slider.offset_top = -34
	slider.offset_bottom = -14
	slider.step = 1
	slider.value_changed.connect(func(v): _show(int(v)))
	layer.add_child(slider)

	dialog = FileDialog.new()
	dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	dialog.access = FileDialog.ACCESS_FILESYSTEM
	dialog.filters = PackedStringArray(["*.jsonl ; FlySnake3D replay"])
	dialog.current_dir = ProjectSettings.globalize_path("res://").path_join("../results/replays").simplify_path()
	dialog.file_selected.connect(_open)
	layer.add_child(dialog)


func _update_hud() -> void:
	if frames.is_empty():
		return
	var fr: Dictionary = frames[idx]
	var res: Dictionary = header["result"]
	info.text = "\n".join([
		"%s   seed %d   %dx%d" % [res["agent"], int(res["seed"]), W, H],
		"step %d / %d   length %d   food %d" % [int(fr["step"]), frames.size() - 1,
			fr["body"].size(), int(fr["food_eaten"])],
		"decision: %s" % fr.get("reason", "-"),
		"%s   %.1f steps/s   camera: %s" % ["playing" if playing else "paused", fps, CAM_NAMES[cam_mode]],
	])
	var outcome := str(fr["outcome"])
	if outcome != "RUNNING":
		status.text = "END: " + outcome
		status.add_theme_color_override("font_color", COL_TRAP if outcome in ["SELF", "WALL", "OBSTACLE"] else Color.WHITE)
	elif first_trap >= 0 and idx >= first_trap:
		status.text = "TRAPPED since step %d: reachable area < body length (%d moves to death)" % [
			first_trap, int(res["steps"]) - first_trap]
		status.add_theme_color_override("font_color", COL_TRAP)
	else:
		status.text = ""


# ------------------------------------------------------------------ input / loop
func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_SPACE:
				if not e.echo:
					playing = not playing
					if playing and idx >= frames.size() - 1:
						_show(0)
			KEY_RIGHT:
				playing = false
				_show(idx + 1)
			KEY_LEFT:
				playing = false
				_show(idx - 1)
			KEY_UP:
				fps = min(fps * 2.0, 1920.0)
			KEY_DOWN:
				fps = max(fps / 2.0, 0.5)
			KEY_HOME:
				_show(0)
			KEY_END:
				_show(frames.size() - 1)
			KEY_T:
				if first_trap >= 0:
					playing = false
					_show(first_trap)
			KEY_1, KEY_2, KEY_3, KEY_4:
				cam_mode = e.keycode - KEY_1
				_update_camera(1.0)
			KEY_O:
				dialog.popup_centered_ratio(0.7)
			KEY_B:
				get_tree().change_scene_to_file("res://brain.tscn")
			KEY_H:
				show_help = not show_help
				help.visible = show_help
		_update_hud()
	elif e is InputEventMouseButton:
		if e.button_index == MOUSE_BUTTON_LEFT:
			dragging = e.pressed
		elif e.button_index == MOUSE_BUTTON_WHEEL_UP and e.pressed:
			orbit_dist = max(orbit_dist * 0.9, 5.0)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN and e.pressed:
			orbit_dist = min(orbit_dist * 1.1, 400.0)
	elif e is InputEventMouseMotion and dragging and cam_mode == Cam.ORBIT:
		orbit_yaw -= e.relative.x * 0.008
		orbit_pitch = clampf(orbit_pitch + e.relative.y * 0.008, 0.15, 1.5)


func _process(delta: float) -> void:
	if playing and not frames.is_empty():
		acc += delta * fps
		var n := int(acc)
		if n > 0:
			acc -= n
			_show(idx + n)
			if idx >= frames.size() - 1:
				playing = false
				_update_hud()
	_update_camera(delta)
