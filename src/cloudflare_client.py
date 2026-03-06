import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class CloudflareClient:
    def __init__(self, worker_url: str):
        self.worker_url = worker_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        raw_body: Optional[bytes] = None,
        content_type: Optional[str] = None,
        expected_statuses: tuple[int, ...] = (200, 201),
    ) -> Any:
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"{self.worker_url}{path}"

        headers: Dict[str, str] = {}
        body: Optional[bytes] = None

        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif raw_body is not None:
            body = raw_body
            headers["Content-Type"] = content_type or "application/octet-stream"

        req = urllib.request.Request(url=url, method=method.upper(), headers=headers, data=body)

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = int(resp.status)
                data = resp.read()
                if status not in expected_statuses:
                    raise RuntimeError(f"Cloudflare API unexpected status {status}: {data.decode('utf-8', 'ignore')}")
                ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    return json.loads(data.decode("utf-8") or "{}")
                return data
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(f"Cloudflare API error {exc.code} for {method} {path}: {payload}") from exc

    def register_heartbeat(self, mac_client_id: str, hostname: str):
        return self._request(
            "POST",
            "/clients/heartbeat",
            json_body={"mac_client_id": mac_client_id, "hostname": hostname},
        )

    def claim_job(self, mac_client_id: str) -> Optional[dict]:
        payload = self._request(
            "POST",
            "/jobs/claim-next",
            json_body={"mac_client_id": mac_client_id},
        )
        job = payload.get("job") if isinstance(payload, dict) else None
        return job or None

    def claim_specific_job(self, job_id: str, mac_client_id: str) -> dict:
        return self._request(
            "POST",
            f"/jobs/{job_id}/claim",
            json_body={"mac_client_id": mac_client_id},
            expected_statuses=(200, 409),
        )

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def get_player(self, player_id: str) -> dict:
        return self._request("GET", f"/players/{player_id}")

    def download_player_photos(self, player_id: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Primary reference photo
        photo_bytes = self._request("GET", f"/players/{player_id}/photo", expected_statuses=(200,))
        primary_path = dest_dir / "reference.jpg"
        primary_path.write_bytes(photo_bytes)

        # Gallery photos → dest_dir/gallery/
        gallery_dir = dest_dir / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        try:
            gallery = self._request("GET", f"/players/{player_id}/gallery", expected_statuses=(200,))
            if isinstance(gallery, list):
                for i, item in enumerate(gallery):
                    file_name = item.get("file", f"{i}.jpg")
                    url_path = f"/players/{player_id}/gallery/{file_name}"
                    try:
                        img_bytes = self._request("GET", url_path, expected_statuses=(200,))
                        (gallery_dir / file_name).write_bytes(img_bytes)
                    except Exception:
                        pass  # skip individual failures
        except Exception:
            pass  # gallery is optional

        return primary_path

    def get_player_embedding(self, player_id: str) -> Optional[List[float]]:
        payload = self._request("GET", f"/players/{player_id}/embedding")
        if not isinstance(payload, dict):
            return None
        emb = payload.get("embedding")
        if isinstance(emb, list):
            return [float(v) for v in emb]
        return None

    def upsert_player_embedding(self, player_id: str, embedding: List[float], metadata: dict):
        return self._request(
            "PUT",
            f"/players/{player_id}/embedding",
            json_body={"embedding": embedding, "metadata": metadata},
        )

    def upload_highlight(self, clip_id: str, video_path: Path) -> str:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Highlight not found: {video_path}")

        ctype = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
        payload = self._request(
            "PUT",
            f"/clips/{clip_id}/video",
            raw_body=video_path.read_bytes(),
            content_type=ctype,
        )

        if isinstance(payload, dict) and "video_url" in payload:
            return str(payload["video_url"])

        return f"{self.worker_url}/clips/{clip_id}/video"

    def complete_job(self, job_id: str, clips: List[dict]):
        return self._request("POST", f"/jobs/{job_id}/complete", json_body={"clips": clips})

    def fail_job(self, job_id: str, error: str):
        return self._request("POST", f"/jobs/{job_id}/fail", json_body={"error": error})
