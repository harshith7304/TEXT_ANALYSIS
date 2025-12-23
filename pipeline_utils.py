from font_normalizer import normalize_font_and_weight

def normalize_analysis_data(analysis_data):
    """
    Takes a list of Gemini analysis results and applies font normalization.
    
    Args:
        analysis_data (list): List of dicts containing 'crop' and 'analysis' keys.
        
    Returns:
        list: The same data structure with 'normalized_font' and 'normalized_weight' added.
    """
    normalized_data = []
    
    for item in analysis_data:
        analysis = item.get("analysis", {})
        
        primary_font = analysis.get("primary_font", "")
        fallback_font = analysis.get("fallback_font", "")
        visual_weight = analysis.get("font_weight", "regular")
        
        # Normalize
        final_font, final_weight = normalize_font_and_weight(
            primary_font, 
            fallback_font, 
            visual_weight
        )
        
        # Create new entry with normalized fields
        new_item = item.copy()
        new_item["analysis"]["normalized_font"] = final_font
        new_item["analysis"]["normalized_weight"] = final_weight
        
        normalized_data.append(new_item)
        
    return normalized_data
