# Robot Cuts Hair

A simulated xArm7 robot arm gives a seated person a haircut, with scissors worked by its gripper and a comb that lifts the hair.

- **Live demo:** https://robotcuts.hair

## What's real and what's simulated

| Part | How it works |
|------|--------------|
| Arm motion | Real output of the DimOS motion planner (RoboPlan) for a mock xArm7, planned ahead of time with the scanned head and neck and the shoulders as collision obstacles. The site and video replay those plans. DimOS isn't running on the server. |
| Scissors | A design that hasn't been built. A ring goes around each gripper finger, so the gripper closing is what closes the blades. The blade angle is solved from the fingers' modelled positions, at a speed within the arm model's limit. A rail fixed to the gripper guides the pivot. The planner doesn't know about the scissors or comb, so `haircut/verify_plans.py` checks their clearance separately. |
| Process | The arm only moves sideways above the hair, because scissors dragged through hair would snag it. For each snip it moves over, goes straight down with the comb into the hair, lifts straight up to the cut height, stops, and closes the blades. |
| Hair and cutting | Physics in the browser, with no scripted motion. A strand stays still until the tool or moving hair reaches it, then acts as a chain that keeps its length and curl and bends under gravity by real hair's stiffness. It can't pass through the scalp, comb, blades, rail, or gripper. Hair passes between the comb's tines going down and rides on them coming up. A strand is cut only where it rises through the blade plane or lies on the comb, and a closing edge crosses it. Cutting force pushes back on the gripper. |
| Hair-to-hair | A voxel grid tracks how crowded each spot is and how its hair moves. Hair is pushed out of crowds, drags on nearby hair and wakes it. Strands don't collide one by one, so they can still pass through each other. |
| Tangling | Moved sideways, the comb and blades catch strands. A caught strand goes taut, then slides along the tool or snaps at about 0.8 N, and can knot with hair it's dragged through. The pull adds up on the arm, which holds still above 20 N until the hair gives. Cut or snapped pieces fall from the speed they had, and land on the comb, in the hair, on the cape, chair or floor. |
| Forces | Tool load and gripper force come from the hair model. Joint figures are holding torques from the URDF's masses and the hair's pull, with no dynamics. The cutting force per hair, scissors' weight, pull limit and snag rates are estimates. |
| Driving by hand | "Drive it yourself" uses inverse kinematics in the page, not DimOS. It keeps the tool 3 mm off the scalp but doesn't check the rest of the arm. Its move back to ready isn't checked by DimOS either. |
| Person and chair | Built in the page, with the scan for the head and neck. The planner knows only the head, neck and shoulders; a separate check found no arm joint or wrist point below the shoulders over the chair. |
| Hairstyles | Five rest shapes grown in the page. Picking one starts a fresh, uncut head. Long hair rests on the cape and chair. |
| Live readout | The "Inside the program" panel and the on-screen overlays read from the running simulation: joint angles, tool position, gripper, current plan and sample, hair contacts, cuts, and a log. |

## Run it locally

The site only needs its built files:

```bash
cd web && bun install && bun run build && python3 -m http.server 8123 --directory dist
```

Publish to the server (`rsync` to `/var/www/robotcuts.hair`, served by Caddy):

```bash
web/deploy.sh
```

## Set up DimOS (only for re-planning)

The site doesn't need this. To plan new moves, clone DimOS next to the `haircut` folder and install it. These plans were made with commit `73bed0714b`:

```bash
git clone https://github.com/dimensionalOS/dimos.git && git -C dimos checkout 73bed0714b && cd dimos && uv sync
```

## Re-plan the arm's moves

```bash
# Terminal 1 (leave running): mock xArm7 + planner, Viser view at http://127.0.0.1:8095
cd haircut && ./py haircut_stack.py

# Terminal 2: plan every move, then check it
cd haircut && ./py plan_library.py --out ../web/public/plans.json
cd haircut && ./py plan_library.py --out ../web/public/plans.json --drag-only
cd haircut && uv run --no-project --python 3.12 --with numpy --with scipy --with trimesh python verify_plans.py ../web/public assets/head_world.stl
```

`./py` runs Python from `dimos/.venv` with the `haircut` folder importable. Run `./py head_model.py` first if the head's placement changes.

## Files

| File | What it does |
|------|--------------|
| `web/src/main.js` | The whole scene: arm, scissors, head, strand hair, cutting, controls |
| `haircut/head_model.py` | Places the head scan at life size, builds the scalp map, and computes the tool paths over the real scalp |
| `haircut/plan_library.py` | Plans every move in DimOS and saves `web/public/plans.json` |
| `haircut/verify_plans.py` | Independent check of clearance to the real head mesh and of joins between moves |
| `haircut/haircut_module.py` | `ManipulationModule` subclass exposing DimOS's internal multi-waypoint Cartesian planner as an RPC |
| `haircut/haircut_stack.py` | The stock `xarm7-planner-coordinator` blueprint with that module swapped in |
| `haircut/export_robot.py` | Exports the xArm7 meshes and joints for the browser |
| `haircut/record.py` | Records a page with headless Chrome and ffmpeg |
| `haircut/haircut.py`, `clipper_path.py` | The first version on a sphere; `clipper_path.py` is still used for the tool's orientation |
| `web/deploy.sh` | Builds the site and publishes it to the server |

## Key numbers

- **Rows:** 5 rows, 2.5 cm apart, each with 7-11 snip stations 2.5 cm apart. The tool tilts 60% of the way from vertical toward the scalp's normal, because the arm can't reach the outer rows fully normal.
- **Speeds:** up to 20 cm/s above the hair and on the way down, 4 cm/s lifting.
- **Cut heights:** 2.5, 3 and 4 cm. At 2 cm the comb has almost no room to go below the cut height, so that setting was dropped.
- **Measured at the 2.5 cm setting (`verify_plans.py`):** blades close 2.41-2.50 cm off the scalp, scissors and comb at least 0.50 cm, fingertips at least 0.60 cm. The tool rises until it's 7.5 cm off the scalp before moving sideways.
- **Cut amount, full haircut at 2.5 cm:** Combed back 35%, Side part 16%, Wavy 38%, Long 11%, Spiky 53% of strands.

## Credits

- Head: "Infinite, 3D Head Scan" by Lee Perry-Smith, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), based on a work at triplegangers.com, obtained from [three.js](https://github.com/mrdoob/three.js). Hair and scalp shading added.
- Robot model: xArm by UFACTORY, BSD licence (`web/public/robot/XARM_LICENSE.txt`).
- Motion planning: [DimOS](https://github.com/dimensionalOS/dimos).
