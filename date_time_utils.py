import pandas as pd


def change_time_zone(
    date_time_utc: [str, pd.Timestamp],
    timezone
) -> pd.Timestamp:
    """
    Changes time from UTC to defined timezone.
    Args:
        date_time_utc: original date time object in UTC
        timezone: time zone to transform the original date time object

    Returns:
        date_time_local: date time object in local time zone
    """

    date_time_local = (pd.to_datetime(date_time_utc)
                       .tz_localize('UTC')
                       .astimezone(timezone)
                       .strftime("%Y-%m-%d, %H:%M"))

    return date_time_local
