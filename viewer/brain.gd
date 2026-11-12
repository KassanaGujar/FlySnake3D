extends Node3D
## Whole-brain activity view: every FlyWire v783 neuron drawn at its real
## position (one anchor point per neuron), lighting up when it spikes.
##
## Plays recordings made by experiments/record_brain_activity.py. Every spike
## shown was produced by the LIF model in Python; nothing here is animated or
## invented.
##
## Command line (after "--"):
##   --brain[=<recording dir>]  default results/brain/looming_demo
##   --check                    headless: load + play all bins, print BRAIN_CHECK, quit
##   --screenshot=<png> --bin=<n> --view=<1-3>
##
## Keys: Space play/pause, Left/Right step 2 ms, Up/Down speed, Home,
##       1 behind / 2 top / 3 side / 4 front view, drag to rotate, wheel to zoom,
##       F fade silent neurons, L labels, G game replay viewer, H help.

const FIRE := Color(4.0, 3.2, 1.2)          # HDR so it glows
const BASE_DIM := 0.45                      # silent neurons stay well below the firing glow
const DECAY := 0.72                         # per 2 ms bin
const CLASS_COLORS := {
	"optic": Color(0.10, 0.16, 0.30),
	"visual_projection": Color(0.10, 0.28, 0.38),
	"visual_centrifugal": Color(0.12, 0.25, 0.32),
	"central": Color(0.22, 0.14, 0.30),
	"descending": Color(0.40, 0.18, 0.18),
	"ascending": Color(0.30, 0.22, 0.14),
	"sensory": Color(0.14, 0.24, 0.16),
	"sensory_ascending": Color(0.16, 0.26, 0.18),
	"motor": Color(0.45, 0.22, 0.10),
	"endocrine": Color(0.30, 0.30, 0.14),
}

var meta: Dictionary = {}
var act: Dictionary = {}
var pos: PackedFloat32Array
var cls: PackedByteArray
var spk_idx: PackedInt32Array
var spk_off: PackedInt32Array
var base_col: PackedColorArray
var glow := {}                # neuron index -> intensity
var n := 0
var bin_i := 0
var playing := false
var bins_per_s := 25.0        # 25 bins/s x 2 ms = 50 ms brain time per second
var acc := 0.0
var fade_silent := false

var yaw := 0.0
var pitch := 0.15
var dist := 900.0
var dragging := false
var view := 1

var mm_inst: MultiMeshInstance3D
var camera: Camera3D
var labels_root: Node3D
var info: Label
var stim_label: Label
var help: Label
var slider: HSlider
var label_nodes := {}          # label text -> Label3D
var show_labels := true
var group_counts := {}        # label text -> recent spike count (decaying)


func _ready() -> void:
	var args := _args()
	var results := ProjectSettings.globalize_path("res://").path_join("../results").simplify_path()
	var rec: String = args.get("brain", "true")
	if rec == "true" or rec == "":
		rec = results.path_join("brain/looming_demo")
	elif rec.is_relative_path():
		rec = ProjectSettings.globalize_path("res://").path_join("..").path_join(rec).simplify_path()
	_build_world()
	_build_hud()
	var err := _load(results.path_join("brain"), rec)
	if err != "":
		push_error(err)
		info.text = err + "\nRun:  python -m experiments.record_brain_activity"
		if args.has("check"):
			print("BRAIN_CHECK " + JSON.stringify({"ok": false, "error": err}))
			get_tree().quit(1)
		return
	_build_brain()
	_build_labels()
	_show(0)
	if args.has("check"):
		var total := 0
		for b in int(act["n_bins"]):
			_show(b)
			total += spk_off[b + 1] - spk_off[b]
		var ok := total == int(act["total_spiking_neuron_bins"]) and n == int(meta["n"])
		print("BRAIN_CHECK " + JSON.stringify({"ok": ok, "neurons": n, "bins": int(act["n_bins"]),
			"spiking_neuron_bins": total}))
		get_tree().quit(0 if ok else 1)
		return
	if args.has("screenshot"):
		_set_view(int(args.get("view", "1")))
		var target := int(args.get("bin", "100"))
		for b in range(maxi(0, target - 12), target + 1):
			_show(b)
		help.visible = false
		_shot(args["screenshot"])
		return
	playing = true


func _args() -> Dictionary:
	var d := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv := a.substr(2).split("=", true, 1)
			d[kv[0]] = kv[1] if kv.size() > 1 else "true"
	return d


func _load(brain_dir: String, rec: String) -> String:
	for f in ["neurons.json", "neurons_pos.f32", "neurons_class.u8"]:
		if not FileAccess.file_exists(brain_dir.path_join(f)):
			return "Missing %s" % brain_dir.path_join(f)
	for f in ["activity.json", "spikes_idx.i32", "spikes_off.i32"]:
		if not FileAccess.file_exists(rec.path_join(f)):
			return "Missing %s" % rec.path_join(f)
	meta = JSON.parse_string(FileAccess.get_file_as_string(brain_dir.path_join("neurons.json")))
	act = JSON.parse_string(FileAccess.get_file_as_string(rec.path_join("activity.json")))
	pos = FileAccess.get_file_as_bytes(brain_dir.path_join("neurons_pos.f32")).to_float32_array()
	cls = FileAccess.get_file_as_bytes(brain_dir.path_join("neurons_class.u8"))
	spk_idx = FileAccess.get_file_as_bytes(rec.path_join("spikes_idx.i32")).to_int32_array()
	spk_off = FileAccess.get_file_as_bytes(rec.path_join("spikes_off.i32")).to_int32_array()
	n = cls.size()
	if pos.size() != n * 3 or spk_off.size() != int(act["n_bins"]) + 1:
		return "Recording files do not match (neurons %d, positions %d)" % [n, pos.size() / 3]
	return ""


# ------------------------------------------------------------------ scene
func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.02, 0.02, 0.035)
	env.glow_enabled = true
	env.glow_intensity = 1.0
	env.glow_bloom = 0.15
	env.glow_hdr_threshold = 1.0
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	camera = Camera3D.new()
	camera.fov = 45
	camera.far = 10000
	camera.current = true
	add_child(camera)


func _build_brain() -> void:
	var mesh := BoxMesh.new()
	mesh.size = Vector3(2.2, 2.2, 2.2)
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mesh.material = mat
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.mesh = mesh
	mm.instance_count = n
	var names: Array = meta["classes"]
	base_col = PackedColorArray()
	base_col.resize(n)
	for i in n:
		var c := int(cls[i])
		var col := Color(0, 0, 0, 0)
		var t := Transform3D(Basis(), Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]))
		if c == 255:
			t = t.scaled(Vector3.ZERO)       # no position in the annotations
		else:
			col = CLASS_COLORS.get(names[c], Color(0.2, 0.2, 0.2)) * BASE_DIM
		base_col[i] = col
		mm.set_instance_transform(i, t)
		mm.set_instance_color(i, col)
	mm_inst = MultiMeshInstance3D.new()
	mm_inst.multimesh = mm
	add_child(mm_inst)


func _build_labels() -> void:
	labels_root = Node3D.new()
	add_child(labels_root)
	for l in meta["labels"]:
		var lab := Label3D.new()
		lab.text = l["text"]
		var p: Array = l["pos"]
		lab.position = Vector3(p[0], p[1], p[2]) + Vector3(0, 14, 0)
		lab.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		lab.no_depth_test = true
		lab.fixed_size = true
		lab.pixel_size = 0.0009
		lab.font_size = 22
		lab.outline_size = 8
		lab.modulate = Color(0.85, 0.95, 1.0)
		labels_root.add_child(lab)
		lab.visible = false
		label_nodes[l["text"]] = lab
		group_counts[l["text"]] = 0.0


func _base(i: int) -> Color:
	var c := base_col[i]
	return Color(c.r, c.g, c.b, 0) * (0.15 if fade_silent else 1.0)


# ------------------------------------------------------------------ playback
func _show(b: int) -> void:
	var nb := int(act["n_bins"])
	b = clampi(b, 0, nb - 1)
	var mm := mm_inst.multimesh
	if b != bin_i + 1:
		# Jumped: rebuild the glow from the preceding bins so the picture
		# matches continuous playback.
		for i in glow:
			mm.set_instance_color(i, _base(i))
		glow.clear()
		for k in group_counts:
			group_counts[k] = 0.0
		for bb in range(maxi(0, b - 12), b):
			_advance(bb, false)
	_advance(b, true)
	bin_i = b
	_update_hud()
	slider.set_value_no_signal(b)


func _advance(b: int, paint: bool) -> void:
	var mm := mm_inst.multimesh
	var dead: Array = []
	for i in glow:
		glow[i] *= DECAY
		if glow[i] < 0.04:
			dead.append(i)
	for i in dead:
		glow.erase(i)
		mm.set_instance_color(i, _base(i))
	for k in range(spk_off[b], spk_off[b + 1]):
		glow[spk_idx[k]] = 1.0
	for i in glow:
		mm.set_instance_color(i, _base(i).lerp(FIRE, glow[i]))
	# group spike counts for the HUD (decaying ~20 ms window)
	var fired := {}
	for k in range(spk_off[b], spk_off[b + 1]):
		fired[spk_idx[k]] = true
	for l in meta["labels"]:
		var c := 0
		for i in l["indices"]:
			if fired.has(int(i)):
				c += 1
		group_counts[l["text"]] = group_counts[l["text"]] * 0.9 + c


# ------------------------------------------------------------------ HUD
func _mk_label(size: int) -> Label:
	var l := Label.new()
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_constant_override("outline_size", 5)
	l.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.9))
	return l


func _build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	info = _mk_label(15)
	info.position = Vector2(14, 10)
	layer.add_child(info)
	stim_label = _mk_label(20)
	stim_label.set_anchors_preset(Control.PRESET_BOTTOM_LEFT)
	stim_label.position = Vector2(14, -74)
	layer.add_child(stim_label)
	help = _mk_label(14)
	help.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	help.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	help.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	help.position = Vector2(-14, 10)
	help.text = "\n".join([
		"Space  play / pause", "Left/Right  step 2 ms", "Up/Down  speed", "Home  restart",
		"1 / 2 / 3 / 4  behind / top / side / front", "drag  rotate,  wheel  zoom",
		"F  fade silent neurons", "L  labels (shown while firing)", "G  game replays", "H  hide help", "",
		"each dot = one neuron at its FlyWire position",
		"bright yellow = spiked in the last few ms",
		"blue = optic lobe, purple = central brain,",
		"red = descending (brain -> body)"])
	layer.add_child(help)
	slider = HSlider.new()
	slider.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	slider.offset_left = 14
	slider.offset_right = -14
	slider.offset_top = -34
	slider.offset_bottom = -14
	slider.step = 1
	slider.max_value = 1
	slider.value_changed.connect(func(v): playing = false; _show(int(v)))
	layer.add_child(slider)


func _update_hud() -> void:
	if act.is_empty():
		return
	slider.max_value = int(act["n_bins"]) - 1
	var b: Dictionary = act["bins"][bin_i]
	var lines := [
		str(act["title"]),
		"%d neurons   t = %.0f ms   %s   %.0fx slow motion" % [n, float(b["t_ms"]),
			"playing" if playing else "paused", 1000.0 / (bins_per_s * float(act["bin_ms"]))],
		"spiking now: %d   glowing: %d   view: %s" % [int(b["n_spiking"]), glow.size(),
			["", "from behind (fly's left = your left)", "from above", "from the right side",
			"from the front (fly's left = your right)"][view]],
		"most active types: " + ", ".join(b["top_types"].map(func(x): return "%s %d" % [x[0], int(x[1])])),
		"",
		"labelled groups (recent spikes):",
	]
	for l in meta["labels"]:
		var v: float = group_counts[l["text"]]
		# a label is shown only while its group is firing, so labels do not pile up
		label_nodes[l["text"]].visible = show_labels and v >= 0.5
		lines.append("  %-30s %s" % [l["text"], "|".repeat(mini(int(round(v)), 40))])
	info.text = "\n".join(lines)
	var on: Array = b["stimuli"]
	if on.is_empty():
		stim_label.text = "Input: none"
		stim_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.75))
	else:
		var names: Array = []
		for k in on:
			var s: Dictionary = act["stimuli"][int(k)]
			names.append("%s at %d Hz" % [s["label"], int(s["rate_hz"])])
		stim_label.text = "Input: " + "; ".join(names)
		stim_label.add_theme_color_override("font_color", Color(1.0, 0.85, 0.4))


# ------------------------------------------------------------------ camera / input
func _set_view(v: int) -> void:
	match v:
		1: yaw = 0.0; pitch = 0.15          # from behind: fly's left is on screen left
		2: yaw = 0.0; pitch = 1.45          # from above, front of the brain at the top
		3: yaw = PI / 2; pitch = 0.1         # from the fly's right side
		4: yaw = PI; pitch = 0.15           # from the front: fly's left is on screen right
	view = v


func _update_camera() -> void:
	camera.position = Vector3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)) * dist
	camera.look_at(Vector3.ZERO, Vector3.UP)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_SPACE:
				if not e.echo:
					playing = not playing
					if playing and bin_i >= int(act.get("n_bins", 1)) - 1:
						_show(0)
			KEY_RIGHT:
				playing = false
				_show(bin_i + 1)
			KEY_LEFT:
				playing = false
				_show(bin_i - 1)
			KEY_UP:
				bins_per_s = minf(bins_per_s * 2.0, 500.0)
			KEY_DOWN:
				bins_per_s = maxf(bins_per_s / 2.0, 1.0)
			KEY_HOME:
				_show(0)
			KEY_1, KEY_2, KEY_3, KEY_4:
				_set_view(e.keycode - KEY_0)
			KEY_F:
				fade_silent = not fade_silent
				for i in n:
					if not glow.has(i):
						mm_inst.multimesh.set_instance_color(i, _base(i))
			KEY_L:
				show_labels = not show_labels
			KEY_H:
				help.visible = not help.visible
			KEY_G:
				Engine.set_meta("left_brain", true)
				get_tree().change_scene_to_file("res://main.tscn")
		_update_hud()
	elif e is InputEventMouseButton:
		if e.button_index == MOUSE_BUTTON_LEFT:
			dragging = e.pressed
		elif e.button_index == MOUSE_BUTTON_WHEEL_UP and e.pressed:
			dist = maxf(dist * 0.9, 100.0)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN and e.pressed:
			dist = minf(dist * 1.1, 5000.0)
	elif e is InputEventMouseMotion and dragging:
		yaw -= e.relative.x * 0.006
		pitch = clampf(pitch + e.relative.y * 0.006, -1.5, 1.5)


func _process(delta: float) -> void:
	if playing and not act.is_empty():
		acc += delta * bins_per_s
		var k := int(acc)
		if k > 0:
			acc -= k
			for j in k:
				if bin_i >= int(act["n_bins"]) - 1:
					playing = false
					break
				_show(bin_i + 1)
	_update_camera()


func _shot(path: String) -> void:
	_update_camera()
	for i in 4:
		await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var err := get_viewport().get_texture().get_image().save_png(path)
	print("BRAIN_SCREENSHOT %s %s" % [path, error_string(err)])
	get_tree().quit(0 if err == OK else 1)
