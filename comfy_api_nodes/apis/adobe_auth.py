"""
Adobe IMS OAuth 2.0 Authentication

This module handles OAuth 2.0 client credentials authentication for Adobe services,
specifically for Firefly API access.

Authentication Flow:
1. Store client_id and client_secret as environment variables or settings
2. Request access token from Adobe IMS using client credentials
3. Cache the token until it expires
4. Automatically refresh when needed

Required Scopes:
- openid
- AdobeID
- session
- additional_info
- read_organizations
- firefly_api
- ff_apis

Environment Variables:
- ADOBE_CLIENT_ID: Your Adobe API client ID
- ADOBE_CLIENT_SECRET: Your Adobe API client secret
- ADOBE_API_KEY: Your Adobe API key (x-api-key header)
"""

from __future__ import annotations
import os
import time
import logging
import json
from pathlib import Path
from typing import Optional, Dict
import aiohttp
from pydantic import BaseModel, Field


def _load_firefly_config() -> Dict[str, str]:
    """
    Load Adobe Firefly configuration from firefly_config.json.

    Returns:
        Dictionary with client_id, client_secret, and api_key
    """
    # Try to find config file in ComfyUI root directory
    config_path = Path(__file__).parent.parent.parent / "firefly_config.json"

    if not config_path.exists():
        logging.warning(
            f"firefly_config.json not found at {config_path}. "
            "Please create it with your Adobe credentials."
        )
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return {
                "client_id": config.get("adobe_client_id", ""),
                "client_secret": config.get("adobe_client_secret", ""),
                "api_key": config.get("adobe_api_key", ""),
            }
    except Exception as e:
        logging.error(f"Error loading firefly_config.json: {e}")
        return {}


class AdobeTokenResponse(BaseModel):
    """Response from Adobe IMS token endpoint"""
    access_token: str = Field(..., description="OAuth access token")
    token_type: str = Field(..., description="Token type (usually 'bearer')")
    expires_in: int = Field(..., description="Token lifetime in seconds")


class AdobeAuthManager:
    """
    Manager for Adobe IMS OAuth 2.0 authentication.

    Handles token acquisition, caching, and automatic refresh.
    """

    # Adobe IMS token endpoint
    TOKEN_ENDPOINT = "https://ims-na1.adobelogin.com/ims/token/v3"

    # Required scopes for Firefly API
    SCOPES = [
        "openid",
        "AdobeID",
        "session",
        "additional_info",
        "read_organizations",
        "firefly_api",
        "ff_apis"
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Adobe authentication manager.

        Args:
            client_id: Adobe API client ID (optional, will try config file then env var)
            client_secret: Adobe API client secret (optional, will try config file then env var)
            api_key: Adobe API key for x-api-key header (optional, will try config file then env var)
        """
        # Load from config file first, then environment variables, then parameters
        config = _load_firefly_config()

        self.client_id = client_id or config.get("client_id") or os.getenv("ADOBE_CLIENT_ID")
        self.client_secret = client_secret or config.get("client_secret") or os.getenv("ADOBE_CLIENT_SECRET")
        self.api_key = api_key or config.get("api_key") or os.getenv("ADOBE_API_KEY")

        # Token cache
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0

        # Add buffer time (5 minutes) before expiry to refresh token
        self._expiry_buffer = 300

    def _is_token_valid(self) -> bool:
        """Check if cached token is still valid"""
        if not self._access_token:
            return False

        current_time = time.time()
        return current_time < (self._token_expiry - self._expiry_buffer)

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Get a valid access token, refreshing if necessary.

        Args:
            force_refresh: Force token refresh even if cached token is valid

        Returns:
            Valid OAuth access token

        Raises:
            ValueError: If client credentials are not configured
            Exception: If token request fails
        """
        # Return cached token if still valid
        if not force_refresh and self._is_token_valid():
            return self._access_token

        # Validate credentials
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Adobe client credentials not configured. Set ADOBE_CLIENT_ID and "
                "ADOBE_CLIENT_SECRET environment variables or pass them to AdobeAuthManager."
            )

        # Request new token
        logging.debug("Requesting new Adobe access token...")

        # Prepare form data
        form_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": ",".join(self.SCOPES),
        }

        # Make request
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.TOKEN_ENDPOINT,
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(
                        f"Failed to get Adobe access token (status {resp.status}): {error_text}"
                    )

                response_json = await resp.json()
                token_response = AdobeTokenResponse.model_validate(response_json)

        # Cache token
        self._access_token = token_response.access_token
        self._token_expiry = time.time() + token_response.expires_in

        logging.debug(
            f"Adobe access token acquired, expires in {token_response.expires_in} seconds"
        )

        return self._access_token

    async def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for Firefly API requests.

        Returns:
            Dictionary with Authorization and x-api-key headers
        """
        access_token = await self.get_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        if self.api_key:
            headers["x-api-key"] = self.api_key

        return headers


# Global auth manager instance (singleton pattern)
_global_auth_manager: Optional[AdobeAuthManager] = None


def get_adobe_auth_manager(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AdobeAuthManager:
    """
    Get global Adobe authentication manager instance.

    Args:
        client_id: Adobe API client ID (optional if already configured)
        client_secret: Adobe API client secret (optional if already configured)
        api_key: Adobe API key (optional if already configured)

    Returns:
        Global AdobeAuthManager instance
    """
    global _global_auth_manager

    if _global_auth_manager is None:
        _global_auth_manager = AdobeAuthManager(
            client_id=client_id,
            client_secret=client_secret,
            api_key=api_key,
        )
    elif client_id or client_secret or api_key:
        # Update credentials if provided
        if client_id:
            _global_auth_manager.client_id = client_id
        if client_secret:
            _global_auth_manager.client_secret = client_secret
        if api_key:
            _global_auth_manager.api_key = api_key

    return _global_auth_manager
