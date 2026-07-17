"""
EcoNetMap - Column Configuration
================================
Canonical column-role definitions and settings.txt mapping helpers.

Lets the toolkit work with any plot-based species occurrence dataset by mapping
a user's own column names onto canonical internal roles (plot_id, species_key,
etc.), persisted in settings.txt inside the project workspace folder. Downstream
modules should resolve columns through resolve_column() rather than hardcoding
source-specific names.

Different data files often use different kinds of species identifier (a code,
a scientific name, a common name) -- which kind doesn't matter. What matters is
that the column mapped to 'species_key' in each file contains values that
actually match across the vegetation, taxa, and regional-pool files, so records
can be linked between them. It's on the user to pick columns that line up.

Author: Flemming Skov (fs@ecos.au.dk)
"""

import configparser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SETTINGS_FILENAME = "settings.txt"

# The pointer file lives next to this module (i.e. in the app/repo directory) and
# just records which project workspace folder settings.txt actually lives in. It is
# gitignored: it's machine-local state, not something a public repo should track.
APP_DIR = Path(__file__).resolve().parent
WORKSPACE_POINTER_FILE = APP_DIR / ".econetmap_workspace"


def get_project_base_path() -> Optional[Path]:
    """Read the configured project/workspace folder from the local pointer file."""
    if not WORKSPACE_POINTER_FILE.exists():
        return None
    path_str = WORKSPACE_POINTER_FILE.read_text().strip()
    return Path(path_str) if path_str else None


def set_project_base_path(path: Path) -> None:
    """Record which project/workspace folder this app instance points at."""
    WORKSPACE_POINTER_FILE.write_text(str(path))


def _settings_path() -> Optional[Path]:
    """Full path to settings.txt inside the configured project workspace, if any."""
    base = get_project_base_path()
    return base / SETTINGS_FILENAME if base else None

# Canonical role schema per dataset type. 'species_key' is the one column that
# links records across all three files -- it can hold a code, a scientific name,
# or a common name; only the required roles are mapped, everything else in a
# file is carried through under its original column name.
VEGETATION_ROLES = {
    'required': ['plot_id', 'species_key', 'habitat_type', 'year', 'x', 'y'],
    'optional': [],
}
TAXA_ROLES = {
    'required': ['species_key'],
    'optional': [],
}
REGIONAL_POOL_ROLES = {
    'required': ['species_key', 'x', 'y'],
    'optional': [],
}

ROLE_LABELS = {
    'plot_id': 'Plot / site identifier',
    'species_key': 'Species identifier (must match across files -- code, scientific name, or common name)',
    'year': 'Sampling year',
    'habitat_type': 'Habitat / vegetation type code',
    'x': 'X coordinate',
    'y': 'Y coordinate',
}

# Convenience default guesses used only to pre-select a sensible option in the
# mapping UI (includes legacy NOVANA names so existing users aren't disrupted).
# Never assumed present -- resolve_column() always falls back to None.
DEFAULT_CANDIDATES = {
    'plot_id': ['aktId', 'plot_id', 'plotId'],
    'species_key': [
        'artId', 'species_id', 'taxonId',
        'videnskabeligtNavn', 'species_scientific', 'scientific_name',
        'almindeligtNavn', 'species_common', 'common_name', 'species',
    ],
    'year': ['aarstal', 'year'],
    'habitat_type': ['naturtypeId', 'habitat_type', 'habitat'],
    'x': ['UTMx', 'x', 'longitude', 'lon'],
    'y': ['UTMy', 'y', 'latitude', 'lat'],
}


def all_roles(roles: dict) -> List[str]:
    """Every role name in a schema dict, across required and optional."""
    return roles.get('required', []) + roles.get('optional', [])


def _read_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    settings_path = _settings_path()
    if settings_path and settings_path.exists():
        config.read(settings_path)
    return config


def load_column_mapping(section: str) -> Dict[str, str]:
    """Load a saved column-mapping section (e.g. 'columns_vegetation') from settings.txt"""
    config = _read_config()
    if not config.has_section(section):
        return {}
    return dict(config.items(section))


def save_column_mapping(section: str, mapping: Dict[str, Optional[str]]) -> None:
    """Persist a column-mapping section to settings.txt, preserving all other sections"""
    settings_path = _settings_path()
    if settings_path is None:
        raise RuntimeError("No project workspace configured yet — set up a project on the home page first.")
    config = _read_config()
    if not config.has_section(section):
        config.add_section(section)
    for role, column in mapping.items():
        if column:
            config.set(section, role, column)
        elif config.has_option(section, role):
            config.remove_option(section, role)
    with open(settings_path, 'w') as f:
        config.write(f)


def guess_default_column(role: str, columns: List[str]) -> Optional[str]:
    """Guess a sensible default column for a role from common naming conventions"""
    for candidate in DEFAULT_CANDIDATES.get(role, []):
        if candidate in columns:
            return candidate
    return None


def resolve_column(df: pd.DataFrame, section: str, role: str) -> Optional[str]:
    """Resolve the actual column name mapped to a canonical role, or None if unmapped/absent"""
    mapping = load_column_mapping(section)
    column = mapping.get(role)
    if column and column in df.columns:
        return column
    return None


def find_mapping_collisions(section: str, roles: dict) -> Dict[str, List[str]]:
    """Find source columns mapped to more than one canonical role (would silently lose data)"""
    mapping = load_column_mapping(section)
    by_source: Dict[str, List[str]] = {}
    for role in all_roles(roles):
        source_col = mapping.get(role)
        if source_col:
            by_source.setdefault(source_col, []).append(role)
    return {col: roles_ for col, roles_ in by_source.items() if len(roles_) > 1}


def rename_to_canonical(df: pd.DataFrame, section: str, roles: dict) -> pd.DataFrame:
    """Add canonical role columns to df, copied from their mapped source columns.

    Uses assignment rather than df.rename() so that two roles mapped to the same
    source column each still get their own copy, instead of one silently
    overwriting the other in a rename dict keyed by source column name.
    """
    df = df.copy()
    for role in all_roles(roles):
        source_col = resolve_column(df, section, role)
        if source_col and source_col != role:
            df[role] = df[source_col]
    return df
