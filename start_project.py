from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None) -> None:
    try:
        result = subprocess.run(command, cwd=cwd, check=False)
    except KeyboardInterrupt as exc:
        joined = " ".join(command)
        raise RuntimeError(f"Interrupted while running command: {joined}") from exc

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


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def terminate_pid(pid: int) -> None:
    try:
        # Try a graceful kill first
        os.kill(pid, signal.SIGTERM)
    except Exception:
        # Fallback for Windows if os.kill didn't work
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        except Exception:
            pass


def ensure_backend_port_free(run_dir: Path, port: int = 8000) -> None:
    if not is_port_in_use(port):
        return

    pid_file = run_dir / "backend.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            print(
                f"Port {port} appears in use, attempting to stop PID {pid} from previous run..."
            )
            terminate_pid(pid)
            # wait a short while for process to exit and port to be released
            for _ in range(10):
                time.sleep(0.5)
                if not is_port_in_use(port):
                    print("Port freed.")
                    pid_file.unlink(missing_ok=True)
                    return
            raise RuntimeError(
                f"Port {port} still in use after attempting to stop PID {pid}."
            )
        except ValueError:
            raise RuntimeError(
                f"Invalid PID file at {pid_file}, and port {port} is in use."
            )
    raise RuntimeError(
        f"Port {port} is already in use. Stop the process using it or change the port."
    )


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start backend and frontend for Vehicle Emission Detection."
    )
    parser.add_argument(
        "--mode",
        choices=["foreground", "background"],
        default="foreground",
        help="foreground keeps services attached to this terminal; background detaches and logs to .run/",
    )
    parser.add_argument(
        "--skip-backend-install",
        action="store_true",
        help="Skip backend pip install step.",
    )
    parser.add_argument(
        "--skip-frontend-install",
        action="store_true",
        help="Skip frontend npm install step.",
    )
    return parser.parse_args()


def prepare_environment(
    project_root: Path,
    backend_root: Path,
    frontend_root: Path,
    skip_backend_install: bool,
    skip_frontend_install: bool,
) -> Path:
    backend_python = get_backend_python(backend_root)

    if not backend_python.exists():
        print("Creating backend virtual environment...")
        launcher = find_python_launcher()
        run_command([*launcher, "-m", "venv", ".venv"], cwd=backend_root)
        backend_python = get_backend_python(backend_root)

    ensure_env_file(backend_root / ".env.example", backend_root / ".env")
    ensure_env_file(frontend_root / ".env.example", frontend_root / ".env")

    run_dir = project_root / ".run"
    run_dir.mkdir(exist_ok=True)

    npm_cmd = get_npm_command()
    if not shutil.which(npm_cmd):
        raise RuntimeError("npm was not found. Install Node.js and retry.")

    if skip_backend_install:
        print("Skipping backend dependency installation (--skip-backend-install).")
    else:
        requirements_file = backend_root / "requirements.txt"
        requirements_hash_file = run_dir / "backend_requirements.sha256"
        current_hash = sha256_of_file(requirements_file)
        previous_hash = (
            requirements_hash_file.read_text(encoding="utf-8").strip()
            if requirements_hash_file.exists()
            else ""
        )

        if current_hash != previous_hash:
            print(
                "Installing backend dependencies (requirements changed or first run)..."
            )
            run_command(
                [str(backend_python), "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=backend_root,
            )
            requirements_hash_file.write_text(current_hash, encoding="utf-8")
        else:
            print("Backend dependencies already up to date. Skipping pip install.")

    if skip_frontend_install:
        print("Skipping frontend dependency installation (--skip-frontend-install).")
    elif not (frontend_root / "node_modules").exists():
        print("Installing frontend dependencies...")
        run_command([npm_cmd, "install"], cwd=frontend_root)

    return backend_python


def start_foreground(backend_python: Path, backend_root: Path, frontend_root: Path) -> None:
    npm_cmd = get_npm_command()

    project_root = backend_root.parent
    run_dir = project_root / ".run"
    run_dir.mkdir(exist_ok=True)
    # ensure port is free or attempt to stop previous background process
    ensure_backend_port_free(run_dir, port=8000)

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


def start_background(project_root: Path, backend_python: Path, backend_root: Path, frontend_root: Path) -> None:
    npm_cmd = get_npm_command()
    run_dir = project_root / ".run"
    backend_log = run_dir / "backend.log"
    frontend_log = run_dir / "frontend.log"

    backend_log_handle = backend_log.open("a", encoding="utf-8")
    frontend_log_handle = frontend_log.open("a", encoding="utf-8")

    backend_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": backend_log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(backend_root),
    }
    frontend_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": frontend_log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(frontend_root),
    }

    # ensure port is free or attempt to stop previous background process
    ensure_backend_port_free(run_dir, port=8000)

    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        backend_kwargs["creationflags"] = flags
        frontend_kwargs["creationflags"] = flags
    else:
        backend_kwargs["start_new_session"] = True
        frontend_kwargs["start_new_session"] = True

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
        **backend_kwargs,
    )
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"],
        **frontend_kwargs,
    )

    (run_dir / "backend.pid").write_text(str(backend_proc.pid), encoding="utf-8")
    (run_dir / "frontend.pid").write_text(str(frontend_proc.pid), encoding="utf-8")

    backend_log_handle.close()
    frontend_log_handle.close()

    print("\nProject started in background mode.")
    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("API Docs: http://localhost:8000/docs")
    print(f"Backend PID: {backend_proc.pid}")
    print(f"Frontend PID: {frontend_proc.pid}")
    print(f"Logs directory: {run_dir}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    backend_root = project_root / "backend"
    frontend_root = project_root / "frontend"

    if not backend_root.exists() or not frontend_root.exists():
        raise RuntimeError("Run this script from the project root where backend/ and frontend/ exist.")

    print("Preparing project startup...")

    backend_python = prepare_environment(
        project_root,
        backend_root,
        frontend_root,
        skip_backend_install=args.skip_backend_install,
        skip_frontend_install=args.skip_frontend_install,
    )

    if args.mode == "background":
        start_background(project_root, backend_python, backend_root, frontend_root)
    else:
        start_foreground(backend_python, backend_root, frontend_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled by user.")
        sys.exit(130)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        sys.exit(1)
