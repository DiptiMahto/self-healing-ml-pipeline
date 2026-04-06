"""
test_runner.py - Helper script to run tests and display results cleanly.
Run with: python test_runner.py
"""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

r = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short', '--no-header'],
    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180
)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[:500])
print("Return code:", r.returncode)
if r.returncode == 0:
    print("\nALL TESTS PASSED!")
else:
    print("\nSOME TESTS FAILED - see above.")
