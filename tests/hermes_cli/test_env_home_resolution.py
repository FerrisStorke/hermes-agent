import os
from pathlib import Path
import hermes_constants
import hermes_cli.profiles as profiles


def test_env_home_and_profiles_root_resolution(tmp_path, monkeypatch):
    root = tmp_path / "var_hermes"
    profiles_root = root / "profiles"
    default_home = profiles_root / "default"
    custom_profile = profiles_root / "custom"

    default_home.mkdir(parents=True)
    custom_profile.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("HERMES_DEFAULT_HOME", str(default_home))
    monkeypatch.setenv("HERMES_PROFILES_ROOT", str(profiles_root))

    # Reset cache memo
    hermes_constants._default_hermes_root_memo = None

    assert profiles._get_default_hermes_home() == default_home
    assert profiles._get_profiles_root() == profiles_root
    assert profiles.get_profile_dir("default") == default_home
    assert profiles.get_profile_dir("custom") == custom_profile

    served = profiles.profiles_to_serve(multiplex=True)
    served_dict = dict(served)
    assert served_dict["default"] == default_home
    assert served_dict["custom"] == custom_profile
