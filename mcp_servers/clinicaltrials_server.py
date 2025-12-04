"""
ClinicalTrials.gov MCP Server

This server exposes ClinicalTrials.gov as an MCP tool, allowing agents to:
1. Search for active trials by condition/intervention
2. Fetch trial details including eligibility criteria
3. Filter trials by status, phase, location
"""

import requests
from typing import List, Dict, Optional
from loguru import logger
import time


class ClinicalTrialsServer:
    """MCP Server for ClinicalTrials.gov API access."""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/"
    
    def __init__(self, rate_limit: float = 0.5):
        """
        Initialize ClinicalTrials.gov server.
        
        Args:
            rate_limit: Seconds between requests (default 0.5s = 2 requests/sec)
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        logger.info("ClinicalTrials.gov MCP Server initialized")
    
    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 20,
        status: Optional[str] = "RECRUITING",
        conditions: Optional[List[str]] = None,
        interventions: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Search for clinical trials.
        
        Args:
            query: Search query (e.g., "glioblastoma")
            max_results: Maximum number of results
            status: Trial status (RECRUITING, ACTIVE, COMPLETED, etc.)
            conditions: List of specific conditions
            interventions: List of specific interventions
        
        Returns:
            List of NCT IDs
        """
        self._rate_limit_wait()
        
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
        }
        
        if status:
            params["filter.overallStatus"] = status
        
        if conditions:
            params["query.cond"] = "|".join(conditions)
        
        if interventions:
            params["query.intr"] = "|".join(interventions)
        
        try:
            response = requests.get(f"{self.BASE_URL}studies", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get("studies", [])
            nct_ids = [study.get("protocolSection", {}).get("identificationModule", {}).get("nctId") 
                      for study in studies]
            nct_ids = [nct_id for nct_id in nct_ids if nct_id]  # Filter out None values
            
            logger.info(f"Found {len(nct_ids)} trials for query: {query}")
            return nct_ids
        
        except Exception as e:
            logger.error(f"Error searching ClinicalTrials.gov: {e}")
            return []
    
    def fetch_trial_details(self, nct_ids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for a list of NCT IDs.
        
        Args:
            nct_ids: List of NCT IDs
        
        Returns:
            List of trial dictionaries
        """
        if not nct_ids:
            return []
        
        trials = []
        
        for nct_id in nct_ids:
            self._rate_limit_wait()
            
            try:
                response = requests.get(
                    f"{self.BASE_URL}studies/{nct_id}",
                    params={"format": "json"},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                protocol = data.get("protocolSection", {})
                
                # Identification
                identification = protocol.get("identificationModule", {})
                nct_id = identification.get("nctId", "")
                brief_title = identification.get("briefTitle", "")
                official_title = identification.get("officialTitle", "")
                
                # Status
                status_module = protocol.get("statusModule", {})
                overall_status = status_module.get("overallStatus", "")
                
                # Conditions
                conditions_module = protocol.get("conditionsModule", {})
                conditions = conditions_module.get("conditions", [])
                
                # Interventions
                arms_module = protocol.get("armsInterventionsModule", {})
                interventions = arms_module.get("interventions", [])
                intervention_list = []
                for intervention in interventions:
                    intervention_list.append({
                        "type": intervention.get("type", ""),
                        "name": intervention.get("name", ""),
                        "description": intervention.get("description", ""),
                    })
                
                # Eligibility
                eligibility_module = protocol.get("eligibilityModule", {})
                eligibility_criteria = eligibility_module.get("eligibilityCriteria", "")
                min_age = eligibility_module.get("minimumAge", "")
                max_age = eligibility_module.get("maximumAge", "")
                sex = eligibility_module.get("sex", "")
                
                # Description
                description_module = protocol.get("descriptionModule", {})
                brief_summary = description_module.get("briefSummary", "")
                detailed_description = description_module.get("detailedDescription", "")
                
                # Design
                design_module = protocol.get("designModule", {})
                study_type = design_module.get("studyType", "")
                phases = design_module.get("phases", [])
                
                trials.append({
                    "nct_id": nct_id,
                    "brief_title": brief_title,
                    "official_title": official_title,
                    "status": overall_status,
                    "conditions": conditions,
                    "interventions": intervention_list,
                    "eligibility_criteria": eligibility_criteria,
                    "min_age": min_age,
                    "max_age": max_age,
                    "sex": sex,
                    "brief_summary": brief_summary,
                    "detailed_description": detailed_description,
                    "study_type": study_type,
                    "phases": phases,
                })
            
            except Exception as e:
                logger.warning(f"Error fetching trial {nct_id}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(trials)} trial details")
        return trials
    
    def search_and_fetch(
        self,
        query: str,
        max_results: int = 20,
        status: Optional[str] = "RECRUITING"
    ) -> List[Dict]:
        """
        Convenience method: Search and fetch trial details in one call.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            status: Trial status filter
        
        Returns:
            List of trial dictionaries
        """
        nct_ids = self.search(query, max_results, status)
        return self.fetch_trial_details(nct_ids)


# Example usage
if __name__ == "__main__":
    # Test the server
    server = ClinicalTrialsServer()
    
    # Search for glioblastoma trials
    trials = server.search_and_fetch("glioblastoma", max_results=3)
    
    for i, trial in enumerate(trials, 1):
        print(f"\n{i}. {trial['brief_title']}")
        print(f"   NCT ID: {trial['nct_id']}")
        print(f"   Status: {trial['status']}")
        print(f"   Conditions: {', '.join(trial['conditions'])}")
        print(f"   Interventions: {len(trial['interventions'])} intervention(s)")
        print(f"   Eligibility: {trial['eligibility_criteria'][:200]}...")

