#!/usr/bin/env python3
"""
Sync script to copy files between local and LTS directories.

This script allows bidirectional syncing of cache and results between
the local delphi-explanations directory and the LTS mount.
"""

import argparse
import shutil
from pathlib import Path
from typing import Literal


class SyncConfig:
    """Configuration for sync operations."""
    
    LOCAL_BASE = Path("/home/jeremias/projects/delphi-explanations/results")
    LTS_BASE = Path("/mnt/ssd-1/soar-automated_interpretability/better_explanations/jeremias/delphi/results")
    
    def __init__(self, direction: Literal["to_lts", "from_lts"] = "to_lts"):
        """
        Initialize sync configuration.
        
        Args:
            direction: "to_lts" to copy from local to LTS, "from_lts" to copy from LTS to local
        """
        if direction not in ("to_lts", "from_lts"):
            raise ValueError(f"Direction must be 'to_lts' or 'from_lts', got {direction}")
        
        self.direction = direction
    
    @property
    def source(self) -> Path:
        """Get source path based on direction."""
        return self.LOCAL_BASE if self.direction == "to_lts" else self.LTS_BASE
    
    @property
    def destination(self) -> Path:
        """Get destination path based on direction."""
        return self.LTS_BASE if self.direction == "to_lts" else self.LOCAL_BASE


def sync_directory(source: Path, destination: Path, target: str = "pythiaST", dry_run: bool = False) -> None:
    """
    Sync a directory from source to destination.
    
    Args:
        source: Source directory path
        destination: Destination directory path
        target: Target subdirectory to sync (default: "pythiaST")
        dry_run: If True, only print what would be done without actually copying
    """
    source_path = source / target
    dest_path = destination / target
    
    if not source_path.exists():
        print(f"❌ Source path does not exist: {source_path}")
        return
    
    if dry_run:
        print(f"[DRY RUN] Would copy from: {source_path}")
        print(f"[DRY RUN] To: {dest_path}")
        return
    
    print(f"🔄 Starting sync...")
    print(f"📂 Source: {source_path}")
    print(f"📂 Destination: {dest_path}")
    
    try:
        if dest_path.exists():
            print(f"⚠️  Destination already exists, removing: {dest_path}")
            shutil.rmtree(dest_path)
        
        shutil.copytree(source_path, dest_path)
        print(f"✅ Successfully synced {target}")
    except Exception as e:
        print(f"❌ Error during sync: {e}")
        raise


def sync_cache(source: Path, destination: Path, dry_run: bool = False) -> None:
    """
    Sync the cache subdirectory.
    
    Args:
        source: Source directory path
        destination: Destination directory path
        dry_run: If True, only print what would be done without actually copying
    """
    sync_directory(source, destination, target="cache", dry_run=dry_run)


def sync_latents(source: Path, destination: Path, dry_run: bool = False) -> None:
    """
    Sync specific latent directories.
    
    Args:
        source: Source directory path
        destination: Destination directory path
        dry_run: If True, only print what would be done without actually copying
    """
    latent_dirs = ["100latents", "400latents"]
    
    for latent_dir in latent_dirs:
        source_latents = source / "pythiaST" / latent_dir
        dest_latents = destination / "pythiaST" / latent_dir
        
        if not source_latents.exists():
            print(f"⚠️  {latent_dir} not found at {source_latents}, skipping...")
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would copy {latent_dir} from: {source_latents}")
            print(f"[DRY RUN] To: {dest_latents}")
            continue
        
        print(f"🔄 Syncing {latent_dir}...")
        try:
            if dest_latents.exists():
                shutil.rmtree(dest_latents)
            
            shutil.copytree(source_latents, dest_latents)
            print(f"✅ Successfully synced {latent_dir}")
        except Exception as e:
            print(f"❌ Error syncing {latent_dir}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Sync files between local and LTS directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync cache to LTS
  python sync_cache.py --target cache

  # Sync latents from LTS to local
  python sync_cache.py --direction from_lts --target latents

  # Dry run to see what would be synced
  python sync_cache.py --target cache --dry-run

  # Sync entire pythiaST directory to LTS
  python sync_cache.py --target pythiaST
        """
    )
    
    parser.add_argument(
        "--direction",
        choices=["to_lts", "from_lts"],
        default="to_lts",
        help="Direction of sync: to_lts (local→LTS) or from_lts (LTS→local). Default: to_lts"
    )
    
    parser.add_argument(
        "--target",
        choices=["cache", "latents", "pythiaST"],
        default="cache",
        help="What to sync: cache, latents, or entire pythiaST. Default: cache"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually copying"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = SyncConfig(direction=args.direction)
    
    print(f"📌 Sync Direction: {args.direction}")
    print(f"📦 Target: {args.target}")
    print()
    
    # Perform sync based on target
    if args.target == "cache":
        sync_cache(config.source, config.destination, dry_run=args.dry_run)
    elif args.target == "latents":
        sync_latents(config.source, config.destination, dry_run=args.dry_run)
    elif args.target == "pythiaST":
        sync_directory(config.source, config.destination, target="pythiaST", dry_run=args.dry_run)
    
    if not args.dry_run:
        print("\n✨ Sync completed!")


if __name__ == "__main__":
    main()
