#!/usr/bin/env python3
"""
Test Genesis with GPU acceleration
"""

import genesis as gs
import time

print("🚀 Genesis GPU Test")
print("=" * 60)

# Test CPU backend
print("\n1. Testing CPU backend...")
gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
scene.build()

start = time.time()
for i in range(100):
    scene.step()
cpu_time = time.time() - start
print(f"✅ CPU: 100 steps in {cpu_time:.3f}s ({100/cpu_time:.0f} steps/sec)")

# Clear Genesis
del scene
gs.init()  # Re-initialize

# Test CUDA backend
print("\n2. Testing CUDA (GPU) backend...")
try:
    gs.init(backend=gs.cuda)
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    start = time.time()
    for i in range(100):
        scene.step()
    gpu_time = time.time() - start
    print(f"✅ GPU: 100 steps in {gpu_time:.3f}s ({100/gpu_time:.0f} steps/sec)")
    
    print(f"\n🎉 GPU Speedup: {cpu_time/gpu_time:.1f}x faster!")
    
except Exception as e:
    print(f"❌ GPU backend failed: {e}")

# Test Vulkan backend (alternative GPU)
print("\n3. Testing Vulkan backend...")
try:
    gs.init(backend=gs.vulkan)
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    start = time.time()
    for i in range(100):
        scene.step()
    vulkan_time = time.time() - start
    print(f"✅ Vulkan: 100 steps in {vulkan_time:.3f}s ({100/vulkan_time:.0f} steps/sec)")
    
except Exception as e:
    print(f"❌ Vulkan backend failed: {e}")

print("\n📊 Performance Summary:")
print(f"CPU Backend: {100/cpu_time:.0f} steps/sec")
if 'gpu_time' in locals():
    print(f"CUDA Backend: {100/gpu_time:.0f} steps/sec (recommended)")
if 'vulkan_time' in locals():
    print(f"Vulkan Backend: {100/vulkan_time:.0f} steps/sec")