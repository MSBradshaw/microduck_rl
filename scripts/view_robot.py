#!/usr/bin/env python3
"""Dump / view a MicroDuck model, with body masses, to inspect the mass budget.

  uv run python scripts/view_robot.py                 # stock robot, table only
  uv run python scripts/view_robot.py --view          # + interactive viewer
  uv run python scripts/view_robot.py --sprung --view # sprung-foot variant
  uv run python scripts/view_robot.py --xml /tmp/o.xml  # write flattened XML

In the viewer: double-click a body to select it; its name shows in the panel.
"""
from __future__ import annotations

import argparse

import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--sprung", action="store_true", help="sprung-foot variant")
ap.add_argument("--stiffness", type=float, default=3344.0)
ap.add_argument("--travel", type=float, default=0.012)
ap.add_argument("--view", action="store_true", help="launch the interactive viewer")
ap.add_argument("--xml", type=str, default=None, help="write the flattened XML here")
args = ap.parse_args()

if args.sprung:
    from mjlab_microduck.robot.sprung_foot import make_sprung_foot_spec_fn
    spec = make_sprung_foot_spec_fn(stiffness=args.stiffness, travel=args.travel)()
    label = f"sprung k={args.stiffness:.0f} travel={args.travel*1000:.0f}mm"
else:
    from mjlab_microduck.robot.microduck_constants import get_walk_spec
    spec = get_walk_spec()
    label = "stock (rigid foot)"

# A floor, so the viewer shows it standing rather than floating in void.
spec.worldbody.add_geom(
    name="_view_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[3.0, 3.0, 0.1], pos=[0.0, 0.0, 0.0],
)
model = spec.compile()

rows = []
for i in range(model.nbody):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if n and n != "world":
        rows.append((model.body_mass[i], n, i))
rows.sort(reverse=True)
total = sum(r[0] for r in rows)

print(f"\n{label}   total = {total*1000:.1f} g   ({len(rows)} bodies)\n")
print(f"{'body':34s} {'mass':>9s} {'cum%':>6s}   parent")
print("-" * 72)
cum = 0.0
for mass, n, i in rows:
    cum += mass
    par = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[i])
    print(f"{n:34s} {mass*1000:7.1f} g {100*cum/total:5.1f}%   {par}")

if args.xml:
    with open(args.xml, "w") as f:
        f.write(spec.to_xml())
    print(f"\nwrote {args.xml}")

if args.view:
    import mujoco.viewer
    data = mujoco.MjData(model)
    print("\nviewer: double-click a body to select it; name appears in the panel.")
    mujoco.viewer.launch(model, data)
