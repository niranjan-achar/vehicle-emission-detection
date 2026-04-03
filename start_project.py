from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"Command failed ({result.returncode}): {joined}")


def find_python_launcher() -> list[str]:
    if shutil.which("py"):
        return ["py", "-3"]
    if shutil.which("python"):
        return ["python"]
    raise RuntimeError("Python was not found. Install Python 3.10+ and retry.")


def ensure_env_file(example_path: Path, env_path: Path) -> None:
    if not env_path.exists() and example_path.exists():
        env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {env_path.name} from {example_path.name}")


def get_backend_python(backend_root: Path) -> Path:
    if os.name == "nt":
        return backend_root / ".venv" / "Scripts" / "python.exe"
    return backend_root / ".venv" / "bin" / "python"


def get_npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def prepare_environment(project_root: Path, backend_root: Path, frontend_root: Path) -> Path:
    backend_python = get_backend_python(backend_root)

    if not backend_python.exists():
        print("Creating backend virtual environment...")
        launcher = find_python_launcher()
        run_command([*launcher, "-m", "venv", ".venv"], cwd=backend_root)

    ensure_env_file(backend_root / ".env.example", backend_root / ".env")
    ensure_env_file(frontend_root / ".env.example", frontend_root / ".env")

    print("Installing backend dependencies...")
    run_command([str(backend_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=backend_root)

    npm_cmd = get_npm_command()
    if not shutil.which(npm_cmd):
        raise RuntimeError("npm was not found. Install Node.js and retry.")

    if not (frontend_root / "node_modules").exists():
        print("Installing frontend dependencies...")
        run_command([npm_cmd, "install"], cwd=frontend_root)

    (project_root / ".run").mkdir(exist_ok=True)
    return backend_python


def start_foreground(backend_python: Path, backend_root: Path, frontend_root: Path) -> None:
    npm_cmd = get_npm_command()

    backend_proc = subprocess.Popen(
        [
            str(backend_python),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=str(backend_root),
    )

    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"],
        cwd=str(frontend_root),
    )

    print("\nProject started in foreground mode.")
    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("API Docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop both services.")

    try:
        while True:
            if backend_proc.poll() is not None:
                raise RuntimeError("Backend process stopped unexpectedly.")
            if frontend_proc.poll() is not None:
                raise RuntimeError("Frontend process stopped unexpectedly.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in (frontend_proc, backend_proc):
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        for proc in (frontend_proc, backend_proc):
            if proc.poll() is None:
                proc.kill()


def start_background(project_root: Path, backend_python: Path, backend_root: Path, frontend_root: Path) -> None:
    npm_cmd = get_npm_command()
    run_dir = project_root / ".run"
    backend_log = run_dir / "backend.log"
    frontend_log = run_dir / "frontend.log"

    backend_log_handle = backend_log.open("a", encoding="utf-8")
    frontend_log_handle = frontend_log.open("a", encoding="utf-8")

    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": backend_log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(backend_root),
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    backend_proc = subprocess.Popen(
        [
            str(backend_python),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        **popen_kwargs,
    )

    frontend_popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": frontend_log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(frontend_root),
    }
    if os.name == "nt":
        frontend_popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        frontend_popen_kwargs["start_new_session"] = True

    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"],
        **frontend_popen_kwargs,
    )

    (run_dir / "backend.pid").write_text(str(backend_proc.pid), encoding="utf-8")
    (run_dir / "frontend.pid").write_text(str(frontend_proc.pid), encoding="utf-8")

    backend_log_handle.close()
    frontend_log_handle.close()

    print("\nProject started in background mode.")
    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("API Docs: http://localhost:8000/docs")
    print(f"Backend PID:  {backend_proc.pid}")
    print(f"Frontend PID: {frontend_proc.pid}")
    print(f"Logs: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start backend and frontend for Vehicle Emission Detection.")
    parser.add_argument(
        "--mode",
        choices=["foreground", "background"],
        default="foreground",
        help="foreground keeps logs in current terminal; background detaches and writes logs to .run/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    backend_root = project_root / "backend"
    frontend_root = project_root / "frontend"

    if not backend_root.exists() or not frontend_root.exists():
        raise RuntimeError("Run this script from the project root where backend/ and frontend/ exist.")

    print("Preparing project startup...")

    backend_python = prepare_environment(project_root, backend_root, frontend_root)

    if args.mode == "background":
        start_background(project_root, backend_python, backend_root, frontend_root)
    else:
        start_foreground(backend_python, backend_root, frontend_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        sys.exit(1)
