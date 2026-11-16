"""Minimal length-prefixed RPC shared by the model servers and the sim-side client.

The model servers and the simulator run in different virtualenvs (different
Python / numpy versions), so pickle is avoided: the message skeleton is JSON and
numpy arrays travel as raw buffers. Only stdlib + numpy are required.
"""

from __future__ import annotations

import json
import socket
import struct
import time
import traceback
from typing import Any, Callable

import numpy as np

_ND = "__nd__"


def _encode(obj: Any, bufs: list[bytes]) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        bufs.append(arr.tobytes())
        return {_ND: len(bufs) - 1, "dtype": arr.dtype.str, "shape": list(arr.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _encode(v, bufs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v, bufs) for v in obj]
    return obj


def _decode(obj: Any, bufs: list[bytes]) -> Any:
    if isinstance(obj, dict):
        if _ND in obj:
            arr = np.frombuffer(bufs[obj[_ND]], dtype=np.dtype(obj["dtype"]))
            return arr.reshape(obj["shape"]).copy()
        return {k: _decode(v, bufs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, bufs) for v in obj]
    return obj


def send_msg(sock: socket.socket, obj: Any) -> None:
    bufs: list[bytes] = []
    header = json.dumps(_encode(obj, bufs)).encode("utf-8")
    parts = [struct.pack(">IQ", len(bufs), len(header)), header]
    for buf in bufs:
        parts.append(struct.pack(">Q", len(buf)))
        parts.append(buf)
    sock.sendall(b"".join(parts))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray(n)
    view = memoryview(out)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("socket closed")
        got += k
    return bytes(out)


def recv_msg(sock: socket.socket) -> Any:
    n_bufs, header_len = struct.unpack(">IQ", _recv_exact(sock, 12))
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    bufs = []
    for _ in range(n_bufs):
        (size,) = struct.unpack(">Q", _recv_exact(sock, 8))
        bufs.append(_recv_exact(sock, size))
    return _decode(header, bufs)


def serve(handler: Callable[[dict], dict], host: str, port: int) -> None:
    """Blocking single-threaded server. Handler errors are returned to the caller."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(4)
        print(f"[server] listening on {host}:{port}", flush=True)
        while True:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with conn:
                while True:
                    try:
                        request = recv_msg(conn)
                    except ConnectionError:
                        break
                    try:
                        response = handler(request)
                    except Exception:  # report to the client instead of dying
                        response = {"error": traceback.format_exc()}
                        print(response["error"], flush=True)
                    send_msg(conn, response)


class RpcClient:
    def __init__(self, host: str, port: int, connect_timeout: float = 1800.0):
        deadline = time.time() + connect_timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=10)
                break
            except OSError:
                if time.time() > deadline:
                    raise ConnectionError(f"could not reach policy server at {host}:{port}")
                time.sleep(2.0)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def call(self, request: dict) -> dict:
        send_msg(self.sock, request)
        response = recv_msg(self.sock)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError("policy server error:\n" + response["error"])
        return response

    def close(self) -> None:
        self.sock.close()
