"""
Metadata Analysis Module
Analyzes EXIF data to detect image tampering indicators
"""

import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

def parse_datetime(date_string):
    """Parse datetime string from EXIF data."""
    if not date_string:
        return None
    
    # Common EXIF datetime formats
    formats = [
        '%Y:%m:%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y:%m:%d',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_string), fmt)
        except (ValueError, TypeError):
            continue
    
    return None

def get_exif_data(image_path):
    """Extract EXIF data from image using PIL."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                exif = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif[tag] = value
                return exif
            else:
                return {}
    except Exception as e:
        return {}

def get_file_timestamps(image_path):
    """Get file system timestamps."""
    try:
        stat = os.stat(image_path)
        return {
            'file_created': datetime.fromtimestamp(
                stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_ctime
            ),
            'file_modified': datetime.fromtimestamp(stat.st_mtime),
            'file_accessed': datetime.fromtimestamp(stat.st_atime)
        }
    except Exception:
        return {}

def analyze_metadata(image_path):
    """
    Comprehensive metadata analysis for a single image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with metadata analysis results
    """
    result = {
        'image_path': str(image_path),
        'image_name': os.path.basename(image_path),
        'has_exif': False,
        'exif_data': {},
        'timestamps': {},
        'file_timestamps': {},
        'tampering_indicators': [],
        'tampering_score': 0,
        'metadata_completeness': 0,
        'analysis_summary': {}
    }
    
    # Get file timestamps
    result['file_timestamps'] = get_file_timestamps(image_path)
    
    # Get EXIF data using PIL
    exif_pil = get_exif_data(image_path)
    result['exif_data'] = exif_pil
    result['has_exif'] = len(exif_pil) > 0
    
    # Extract key timestamps from EXIF
    timestamp_fields = {
        'DateTime': 'DateTime',
        'DateTimeOriginal': 'DateTimeOriginal',
        'DateTimeDigitized': 'DateTimeDigitized',
        'EXIF DateTimeOriginal': 'EXIF DateTimeOriginal',
        'EXIF DateTimeDigitized': 'EXIF DateTimeDigitized',
        'Image DateTime': 'Image DateTime'
    }
    
    for key, label in timestamp_fields.items():
        if key in result['exif_data']:
            dt = parse_datetime(result['exif_data'][key])
            if dt:
                result['timestamps'][label] = dt
    
    # Calculate metadata completeness
    expected_fields = ['DateTime', 'DateTimeOriginal', 'Make', 'Model', 'Software']
    found_fields = sum(1 for field in expected_fields if field in result['exif_data'])
    result['metadata_completeness'] = (found_fields / len(expected_fields)) * 100
    
    # Tampering Detection Logic
    tampering_score = 0
    indicators = []
    
    # 1. Check if EXIF data is missing (suspicious)
    if not result['has_exif']:
        tampering_score += 20
        indicators.append("No EXIF data found - image may have been stripped of metadata")
    
    # 2. Check for missing critical timestamps
    if 'DateTimeOriginal' not in result['timestamps']:
        tampering_score += 15
        indicators.append("Missing DateTimeOriginal - original capture time unknown")
    
    # 3. Check for timestamp inconsistencies
    timestamps = list(result['timestamps'].values())
    if len(timestamps) > 1:
        sorted_timestamps = sorted(timestamps)
        if timestamps != sorted_timestamps:
            tampering_score += 25
            indicators.append("Timestamp inconsistencies detected - timestamps not in chronological order")
    
    # 4. Compare EXIF timestamps with file timestamps
    if result['timestamps'] and result['file_timestamps']:
        exif_times = list(result['timestamps'].values())
        file_modified = result['file_timestamps'].get('file_modified')
        
        if file_modified and exif_times:
            latest_exif = max(exif_times)
            if file_modified > latest_exif:
                time_diff = (file_modified - latest_exif).total_seconds() / 3600  # hours
                if time_diff > 1:  # More than 1 hour difference
                    tampering_score += 10
                    indicators.append(f"File modified {time_diff:.1f} hours after EXIF timestamp - possible tampering")
    
    # 5. Check for editing software indicators
    software = result['exif_data'].get('Software', '')
    if software:
        editing_software = ['photoshop', 'gimp', 'lightroom', 'paint', 'editor', 'edit']
        if any(term in str(software).lower() for term in editing_software):
            tampering_score += 15
            indicators.append(f"Editing software detected: {software}")
    
    # 6. Check for suspicious metadata patterns
    if 'Make' in result['exif_data'] and 'Model' in result['exif_data']:
        make = str(result['exif_data']['Make']).lower()
        model = str(result['exif_data']['Model']).lower()
        if 'unknown' in make or 'unknown' in model or make == '' or model == '':
            tampering_score += 10
            indicators.append("Missing or generic camera information")
    
    # 7. Check for GPS data
    has_gps = any('GPS' in key or 'gps' in key.lower() for key in result['exif_data'].keys())
    if not has_gps and result['has_exif']:
        indicators.append("No GPS data found (may indicate AI-generated or processed image)")
    
    result['tampering_score'] = min(tampering_score, 100)  # Cap at 100
    result['tampering_indicators'] = indicators
    
    # Create summary
    result['analysis_summary'] = {
        'likely_tampered': tampering_score >= 50,
        'tampering_confidence': 'High' if tampering_score >= 70 else 'Medium' if tampering_score >= 40 else 'Low',
        'metadata_quality': 'Complete' if result['metadata_completeness'] >= 80 else 'Partial' if result['metadata_completeness'] >= 40 else 'Minimal',
        'estimated_tampering_time': None
    }
    
    # Estimate tampering time
    if result['timestamps'] and result['file_timestamps']:
        file_modified = result['file_timestamps'].get('file_modified')
        if file_modified:
            exif_times = list(result['timestamps'].values())
            if exif_times:
                latest_exif = max(exif_times)
                if file_modified > latest_exif:
                    result['analysis_summary']['estimated_tampering_time'] = file_modified
    
    return result

