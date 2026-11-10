extends RefCounted
## Reads the JSONL replays written by game/runner.py:
##   line 1  {"type": "header", "config": {...}, "result": {...}}
##   line 2+ {"type": "frame", "step", "body", "heading", "food", "food_eaten",
##            "outcome", ["plan"], ["reason"]}


static func load_replay(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {"error": "cannot open %s (%s)" % [path, error_string(FileAccess.get_open_error())]}
	var header = null
	var frames: Array = []
	var line_no := 0
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		line_no += 1
		if line.is_empty():
			continue
		var d = JSON.parse_string(line)
		if typeof(d) != TYPE_DICTIONARY:
			return {"error": "bad JSON at line %d" % line_no}
		if d.get("type") == "header":
			header = d
		else:
			frames.append(d)
	if header == null or frames.is_empty():
		return {"error": "missing header or frames"}
	return {"header": header, "frames": frames}


## Consistency checks used by the headless test. Returns a list of problems.
static func validate(header: Dictionary, frames: Array) -> Array:
	var problems: Array = []
	var res: Dictionary = header["result"]
	if frames.size() != int(res["steps"]) + 1:
		problems.append("frame count %d != steps + 1 (%d)" % [frames.size(), int(res["steps"]) + 1])
	if str(frames[-1]["outcome"]) != str(res["outcome"]):
		problems.append("last outcome %s != header %s" % [frames[-1]["outcome"], res["outcome"]])
	var w := int(header["config"]["width"])
	var h := int(header["config"]["height"])
	for i in frames.size():
		var body: Array = frames[i]["body"]
		if int(frames[i]["step"]) != i:
			problems.append("frame %d has step %d" % [i, int(frames[i]["step"])])
			break
		for k in body.size():
			var c: Array = body[k]
			if c[0] < 0 or c[0] >= w or c[1] < 0 or c[1] >= h:
				problems.append("frame %d: segment out of bounds" % i)
			if k > 0:
				var p: Array = body[k - 1]
				if absi(int(c[0]) - int(p[0])) + absi(int(c[1]) - int(p[1])) != 1:
					problems.append("frame %d: body not contiguous at %d" % [i, k])
		if problems.size() > 5:
			break
	return problems
