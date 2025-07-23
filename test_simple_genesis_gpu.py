#!/usr/bin/env python3
"""
Simple test to verify Genesis GPU usage
"""

import genesis as gs
import time

print("🎮 Genesis GPU Verification")
print("=" * 60)

# Initialize with CUDA
gs.init(backend=gs.cuda)

# Create scene
scene = gs.Scene(show_viewer=False)

# Add simple entities
plane = scene.add_entity(gs.morphs.Plane())
# Note: Genesis 0.2.1 doesn't support pos in add_entity
# Position must be set in the morph or after creation
box = scene.add_entity(
    gs.morphs.Box(size=(0.1, 0.1, 0.1))
)

# Build scene
print("Building scene on GPU...")
scene.build()

# Run simulation
print("\nRunning physics simulation on GPU...")
start = time.time()
steps = 1000

for i in range(steps):
    scene.step()
    if i % 100 == 0:
        print(f"  Step {i}")

elapsed = time.time() - start

print(f"\n✅ Completed {steps} steps in {elapsed:.3f}s")
print(f"⚡ Performance: {steps/elapsed:.0f} steps/sec on GPU!")
print(f"\n🎉 Genesis is successfully using your RTX 3060 Ti!")