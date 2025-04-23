import boto3
import pandas as pd
import numpy as np
import json
import logging
import time
import hashlib
from io import StringIO
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon, mapping, MultiPoint
from shapely.ops import unary_union
import colorsys
import traceback
from functools import lru_cache
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global cache for recent results
# Simple in-memory cache (could be replaced with DynamoDB or ElastiCache for real persistence)
RESULTS_CACHE = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Cache decorator for expensive operations
def timed_lru_cache(seconds=600, maxsize=128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time.time() + seconds
        
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() > func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*args, **kwargs)
        
        return wrapped_func
    
    return wrapper_cache

@timed_lru_cache(seconds=600)
def get_s3_file(bucket, key):
    """Cached retrieval of S3 files to reduce unnecessary downloads"""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return content
    except Exception as e:
        logger.error(f"Error reading {key} from S3 bucket {bucket}: {str(e)}")
        raise

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def degrees_to_km(degrees, latitude=0):
    """
    Convert degrees to kilometers at a given latitude
    Uses the equator if no latitude specified
    """
    # 1 degree of latitude is approx 111km
    # 1 degree of longitude varies with latitude
    km_per_degree = 111
    return degrees * km_per_degree

def get_cluster_color(cluster_id, is_noise=False):
    """Generate visually distinct colors for each cluster"""
    if is_noise:
        return "#808080"  # Gray for noise
    
    # Generate distinct colors using HSV color space
    hue = (cluster_id * 0.618033988749895) % 1.0  # Golden ratio for better distribution
    saturation = 0.7
    value = 0.9
    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"#{r:02x}{g:02x}{b:02x}"

def create_concave_hull(points, alpha=0.5):
    """
    Create a concave hull (alpha shape) from a set of points.
    Falls back to convex hull if concave hull creation fails.
    
    Args:
        points: Array of [lon, lat] coordinates
        alpha: Alpha parameter controlling concavity (smaller = more concave)
    
    Returns:
        Shapely polygon representing the hull
    """
    if len(points) < 4:
        # Not enough points for a concave hull, use convex
        try:
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])
        except Exception as e:
            logger.warning(f"Could not create convex hull: {str(e)}")
            # If even that fails, create a small buffer around points
            return MultiPoint(points).buffer(0.01)
    
    try:
        # Try to create a concave hull using alpha shapes
        # This is a simplified version - more sophisticated alpha shape 
        # algorithms exist but they're beyond the scope of this function
        from scipy.spatial import Delaunay
        
        delaunay = Delaunay(points)
        triangles = points[delaunay.simplices]
        
        # Calculate the circumradius of each triangle
        radii = []
        for triangle in triangles:
            # Calculate side lengths
            a = np.linalg.norm(triangle[0] - triangle[1])
            b = np.linalg.norm(triangle[1] - triangle[2])
            c = np.linalg.norm(triangle[2] - triangle[0])
            
            # Semi-perimeter
            s = (a + b + c) / 2
            
            # Area using Heron's formula
            try:
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                # Circumradius
                radius = (a * b * c) / (4 * area) if area > 0 else float('inf')
                radii.append(radius)
            except:
                radii.append(float('inf'))
        
        radii = np.array(radii)
        
        # Keep only triangles with circumradius less than alpha
        keep = radii < (1 / alpha)
        
        # Create polygons for each triangle to keep
        triangles_to_keep = [Polygon(triangle) for i, triangle in enumerate(triangles) if keep[i]]
        
        if not triangles_to_keep:
            # Fall back to convex hull if no triangles meet criteria
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])
        
        # Combine all triangles
        concave_hull = unary_union(triangles_to_keep)
        
        # Ensure it's a valid polygon
        if not concave_hull.is_valid or concave_hull.is_empty:
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])
            
        return concave_hull
    
    except Exception as e:
        logger.warning(f"Error creating concave hull, falling back to convex hull: {str(e)}")
        try:
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])
        except Exception as e2:
            logger.warning(f"Could not create convex hull either: {str(e2)}")
            # Last resort: buffer around points
            return MultiPoint(points).buffer(0.01)

def lambda_handler(event, context):
    """
    Lambda function to perform DBSCAN clustering on geospatial healthcare data.
   
    Expected parameters in event:
    - eps: Float for DBSCAN epsilon (neighborhood size) (default 0.03)
    - min_samples: Integer for minimum points to form a cluster (default 5)
    - include_disease_points: Boolean to include disease point coordinates in the response (default True)
    - use_concave_hull: Boolean to use concave hull instead of convex hull for clusters (default True)
    - alpha: Float for controlling concavity of hull (default 0.5)
    - use_cache: Boolean to use cached results if available (default True)
   
    Returns:
    - GeoJSON with DBSCAN clusters, hospital locations, and disease points
    """
    # Performance tracking
    start_time = time.time()
    
    # Initialize response elements to track progress in case of failure
    processing_status = {
        "params_parsed": False,
        "files_loaded": False,
        "dbscan_complete": False,
        "hospital_processing_complete": False,
        "disease_processing_complete": False,
        "clusters_created": False
    }
    
    # Parse incoming parameters
    try:
        # Extract parameters with defaults
        params = {
            'eps': 0.03,
            'min_samples': 5,
            'include_disease_points': True,
            'use_concave_hull': True,
            'alpha': 0.5,
            'use_cache': True
        }
       
        if 'body' in event and event['body']:
            # Handle body content from API Gateway
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
           
            # Update parameters from request
            if 'eps' in body:
                params['eps'] = float(body['eps'])
            if 'min_samples' in body:
                params['min_samples'] = int(body['min_samples'])
            if 'include_disease_points' in body:
                params['include_disease_points'] = bool(body['include_disease_points'])
            if 'use_concave_hull' in body:
                params['use_concave_hull'] = bool(body['use_concave_hull'])
            if 'alpha' in body:
                params['alpha'] = float(body['alpha'])
            if 'use_cache' in body:
                params['use_cache'] = bool(body['use_cache'])
        elif any(k in event for k in params.keys()):
            # Direct parameter access
            for key in params.keys():
                if key in event:
                    if key in ['min_samples']:
                        params[key] = int(event[key])
                    elif key in ['include_disease_points', 'use_concave_hull', 'use_cache']:
                        params[key] = bool(event[key])
                    else:
                        params[key] = float(event[key])
           
        # Ensure parameters are within reasonable ranges
        params['eps'] = max(0.001, min(params['eps'], 0.2))
        params['min_samples'] = max(2, min(params['min_samples'], 100))
        params['alpha'] = max(0.1, min(params['alpha'], 2.0))
       
        logger.info(f"Using parameters: {params}")
        processing_status["params_parsed"] = True
    except Exception as e:
        logger.error(f"Error parsing parameters: {str(e)}")
        logger.error(traceback.format_exc())
        # Use defaults if there's an error
        params = {
            'eps': 0.03,
            'min_samples': 5,
            'include_disease_points': True,
            'use_concave_hull': True,
            'alpha': 0.5,
            'use_cache': True
        }
        processing_status["params_parsed"] = "with_defaults"

    # Generate cache key based on parameters
    if params['use_cache']:
        cache_key = hashlib.md5(
            f"{params['eps']}:{params['min_samples']}:{params['include_disease_points']}:{params['use_concave_hull']}:{params['alpha']}".encode()
        ).hexdigest()
        
        # Check if we have a cached result
        if cache_key in RESULTS_CACHE:
            cached_result = RESULTS_CACHE[cache_key]
            cache_time = cached_result['timestamp']
            
            # Check if cache is still valid (within TTL)
            if time.time() - cache_time < CACHE_TTL_SECONDS:
                logger.info(f"Using cached result for key {cache_key}")
                cached_response = cached_result['response']
                
                # Add cache metadata to response
                if isinstance(cached_response.get('body'), str):
                    try:
                        body_data = json.loads(cached_response['body'])
                        body_data['cache'] = {
                            'hit': True,
                            'age_seconds': int(time.time() - cache_time)
                        }
                        cached_response['body'] = json.dumps(body_data)
                    except:
                        # If we can't modify the cache metadata, just return the cached response as is
                        pass
                
                return cached_response

    try:
        logger.info("Starting DBSCAN clustering analysis")
        
        # S3 bucket and file details
        bucket_name = 'for-kde-based-data'
        hospitals_file = 'master database.csv'
        disease_file = 'disease random points.csv'
       
        # Get the hospitals CSV file from S3
        logger.info(f"Fetching hospitals CSV file {hospitals_file} from bucket {bucket_name}")
        try:
            hospitals_content = get_s3_file(bucket_name, hospitals_file)
        except Exception as e:
            logger.error(f"Error reading hospitals CSV: {str(e)}")
            # Try alternative file name with uppercase
            try:
                alt_hospitals_file = 'MASTER DATABASE.csv'
                logger.info(f"Trying alternate file name: {alt_hospitals_file}")
                hospitals_content = get_s3_file(bucket_name, alt_hospitals_file)
            except Exception as e2:
                logger.error(f"Error reading alternate hospitals CSV: {str(e2)}")
                raise Exception(f"Could not read hospital data from S3: {str(e2)}")
       
        # Parse hospitals CSV
        logger.info("Parsing hospitals CSV")
        hospitals_df = pd.read_csv(StringIO(hospitals_content))
        logger.info(f"Successfully loaded hospitals CSV with {len(hospitals_df)} records")
       
        # Get the disease points CSV file from S3
        logger.info(f"Fetching disease points CSV file {disease_file} from bucket {bucket_name}")
        try:
            disease_content = get_s3_file(bucket_name, disease_file)
        except Exception as e:
            logger.error(f"Error reading disease CSV: {str(e)}")
            raise Exception(f"Could not read disease data from S3: {str(e)}")
       
        # Parse disease points CSV
        logger.info("Parsing disease points CSV")
        disease_df = pd.read_csv(StringIO(disease_content))
        logger.info(f"Successfully loaded disease points CSV with {len(disease_df)} records")
        
        processing_status["files_loaded"] = True
       
        # Clean data - handle missing coordinates
        hospitals_df = hospitals_df.dropna(subset=['latitude', 'longitude'])
        disease_df = disease_df.dropna(subset=['latitude', 'longitude'])
        
        # Convert to proper numeric types if not already
        hospitals_df['latitude'] = pd.to_numeric(hospitals_df['latitude'], errors='coerce')
        hospitals_df['longitude'] = pd.to_numeric(hospitals_df['longitude'], errors='coerce')
        disease_df['latitude'] = pd.to_numeric(disease_df['latitude'], errors='coerce')
        disease_df['longitude'] = pd.to_numeric(disease_df['longitude'], errors='coerce')
        
        # Drop any rows where conversion to numeric failed
        hospitals_df = hospitals_df.dropna(subset=['latitude', 'longitude'])
        disease_df = disease_df.dropna(subset=['latitude', 'longitude'])
       
        logger.info(f"After cleaning: {len(hospitals_df)} hospitals, {len(disease_df)} disease points")
       
        if len(disease_df) < params['min_samples']:
            raise ValueError(f"Insufficient disease data points ({len(disease_df)}) for DBSCAN analysis with min_samples={params['min_samples']}")
       
        # Extract coordinates
        hospital_coords = hospitals_df[['longitude', 'latitude']].values
        disease_coords = disease_df[['longitude', 'latitude']].values
       
        # Get map bounds with padding
        padding = 0.02  # Add padding around the bounds
        min_lng = min(disease_coords[:, 0].min(), hospital_coords[:, 0].min()) - padding
        max_lng = max(disease_coords[:, 0].max(), hospital_coords[:, 0].max()) + padding
        min_lat = min(disease_coords[:, 1].min(), hospital_coords[:, 1].min()) - padding
        max_lat = max(disease_coords[:, 1].max(), hospital_coords[:, 1].max()) + padding
       
        # For the user's awareness, convert the epsilon to approximate kilometers
        center_lat = (min_lat + max_lat) / 2
        eps_km = degrees_to_km(params['eps'], center_lat)
        logger.info(f"Epsilon {params['eps']} degrees is approximately {eps_km:.2f} km at latitude {center_lat:.2f}")
        
        # Run DBSCAN directly on original coordinates (this is the key fix)
        logger.info(f"Running DBSCAN with eps={params['eps']}, min_samples={params['min_samples']}")
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        clusters = dbscan.fit_predict(disease_coords)
       
        # Convert clusters to Python list of integers (for proper JSON serialization)
        clusters = [int(c) for c in clusters]
        
        processing_status["dbscan_complete"] = True
       
        # Count unique clusters (excluding noise with label -1)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = clusters.count(-1)
       
        logger.info(f"Found {n_clusters} clusters and {n_noise} noise points")
    
        # Create GeoJSON features for hospitals
        logger.info("Creating GeoJSON for hospital points")
        hospital_features = []
       
        for idx, row in hospitals_df.iterrows():
            try:
                # Ensure all values are proper Python types for JSON
                props = {}
                
                # Map DataFrame columns to GeoJSON properties with proper handling
                column_mapping = {
                    'id': 'id',
                    'hospital/doctor_name': 'name',
                    'hospital_type': 'type',
                    'hospital_category': 'category',
                    'hospital_beds_availability': 'beds',
                    'hospital_contact_number': 'contact',
                    'email': 'email',
                    'hospital_category_specialities': 'specialities',
                    'timings': 'timings'
                }
                
                for df_col, prop_name in column_mapping.items():
                    if df_col in row and pd.notna(row[df_col]):
                        # Ensure proper type conversion
                        if df_col == 'id':
                            props[prop_name] = int(row[df_col])
                        else:
                            props[prop_name] = str(row[df_col])
                    else:
                        props[prop_name] = "Unknown" if prop_name != 'id' else int(idx)
                
                # Create hospital feature
                feature = {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row['longitude']), float(row['latitude'])]
                    }
                }
                hospital_features.append(feature)
            except Exception as e:
                logger.warning(f"Error creating hospital point for idx {idx}: {str(e)}")
        
        processing_status["hospital_processing_complete"] = True
       
        # Create GeoJSON for disease points if requested
        disease_features = []
        if params['include_disease_points']:
            logger.info("Creating GeoJSON for disease points")
            for idx, (coords, cluster_id) in enumerate(zip(disease_coords, clusters)):
                try:
                    is_noise = bool(cluster_id == -1)  # Convert to Python bool
                    disease_type = str(disease_df.iloc[idx]['disease_type']) if 'disease_type' in disease_df.columns else "Unknown"
                    
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "id": int(idx),
                            "cluster_id": cluster_id,  # Already int
                            "is_noise": is_noise,  # Python bool
                            "color": get_cluster_color(cluster_id, is_noise),
                            "disease_type": disease_type
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(coords[0]), float(coords[1])]
                        }
                    }
                    disease_features.append(feature)
                except Exception as e:
                    logger.warning(f"Error creating disease point for idx {idx}: {str(e)}")
        
        processing_status["disease_processing_complete"] = True
        
        # Process each cluster to create polygons
        logger.info("Creating GeoJSON features for clusters")
        cluster_features = []
       
        # Process each cluster
        max_cluster_id = max(clusters) if clusters else -1
        for cluster_id in range(0, max_cluster_id + 1):  # Skip noise (cluster_id = -1)
            # Skip if no points with this cluster id
            if cluster_id not in clusters:
                continue
           
            # Get points in this cluster
            mask = [c == cluster_id for c in clusters]
            if not any(mask):
                continue
               
            # Extract cluster points
            cluster_points = disease_coords[mask]
           
            # Skip clusters with too few points for hull creation
            if len(cluster_points) < 3:
                continue
           
            # Create polygon from hull of points
            try:
                # Choose hull type based on parameters
                if params['use_concave_hull']:
                    # Use concave hull (alpha shape)
                    polygon = create_concave_hull(cluster_points, params['alpha'])
                else:
                    # Use convex hull (traditional)
                    hull = ConvexHull(cluster_points)
                    polygon = Polygon(cluster_points[hull.vertices])
                
                # Add small buffer for smoother polygons
                buffer_distance = min(0.005, params['eps'] / 10)  # Scale buffer based on epsilon
                polygon = polygon.buffer(buffer_distance)
               
                # Skip invalid polygons
                if not polygon.is_valid or polygon.is_empty:
                    logger.warning(f"Generated invalid polygon for cluster {cluster_id}")
                    continue
                   
                # Add to feature collection
                point_count = sum(mask)
                feature = {
                    "type": "Feature",
                    "properties": {
                        "cluster_id": cluster_id,  # Already int
                        "point_count": point_count,  # Already int
                        "color": get_cluster_color(cluster_id),
                        "fill_opacity": 0.5,
                        "area_sq_km": polygon.area * 111 * 111,  # Approximate area in sq km
                        "eps_value": float(params['eps']),
                        "eps_km": float(eps_km)
                    },
                    "geometry": mapping(polygon)
                }
                cluster_features.append(feature)
               
            except Exception as e:
                logger.warning(f"Error creating polygon for cluster {cluster_id}: {str(e)}")
                logger.warning(traceback.format_exc())
        
        processing_status["clusters_created"] = True
       
        # Calculate center point for map
        center_lng = float((min_lng + max_lng) / 2)
        center_lat = float((min_lat + max_lat) / 2)
       
        # Prepare final response
        logger.info("Preparing response")
        response_data = {
            "clusters": {
                "type": "FeatureCollection",
                "features": cluster_features
            },
            "hospitals": {
                "type": "FeatureCollection",
                "features": hospital_features
            },
            "diseases": {
                "type": "FeatureCollection",
                "features": disease_features
            },
            "center": [center_lat, center_lng],
            "bounds": [[float(min_lat), float(min_lng)], [float(max_lat), float(max_lng)]],
            "stats": {
                "disease_count": int(len(disease_df)),
                "hospital_count": int(len(hospitals_df)),
                "cluster_count": int(n_clusters),
                "noise_points": int(n_noise),
                "eps": float(params['eps']),
                "eps_km": float(eps_km),
                "min_samples": int(params['min_samples']),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            },
            "cache": {
                "hit": False
            }
        }
       
        # Create final response object
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',  # Allow CORS for frontend
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'
            },
            'body': json.dumps(response_data, cls=NumpyEncoder)
        }
        
        # Cache the result if caching is enabled
        if params['use_cache']:
            RESULTS_CACHE[cache_key] = {
                'timestamp': time.time(),
                'response': response
            }
            
            # Cleanup old cache entries
            current_time = time.time()
            keys_to_remove = []
            for key, cached_data in RESULTS_CACHE.items():
                if current_time - cached_data['timestamp'] > CACHE_TTL_SECONDS:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del RESULTS_CACHE[key]
        
        logger.info(f"Successfully completed DBSCAN analysis in {(time.time() - start_time) * 1000:.2f}ms")
        return response
   
    except Exception as e:
        # Log detailed error
        error_time = (time.time() - start_time) * 1000
        logger.error(f"Error processing data after {error_time:.2f}ms: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Include processing status in the error response for debugging
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',  # Allow CORS
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Error processing DBSCAN clustering',
                'processing_status': processing_status,
                'processing_time_ms': int(error_time)
            })
        }

# Import needed for the cache decorator
from functools import wraps
