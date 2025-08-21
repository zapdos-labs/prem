#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="http://localhost:8080"
TIMEOUT=30

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}[PASS]${NC} $message"
        ((PASSED_TESTS++))
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}[FAIL]${NC} $message"
        ((FAILED_TESTS++))
    elif [ "$status" = "INFO" ]; then
        echo -e "${BLUE}[INFO]${NC} $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}[WARN]${NC} $message"
    fi
    ((TOTAL_TESTS++))
}

# Function to test if server is running
check_server() {
    echo -e "${BLUE}=== Checking Server Status ===${NC}"
    
    response=$(curl -s -w "%{http_code}" -o /dev/null --connect-timeout $TIMEOUT "$BASE_URL/")
    if [ "$response" = "200" ]; then
        print_status "PASS" "Server is running at $BASE_URL"
        return 0
    else
        print_status "FAIL" "Server is not responding (HTTP: $response)"
        echo -e "${YELLOW}Please make sure your server is running with: python main.py${NC}"
        exit 1
    fi
}

# Test health endpoint
test_health() {
    echo -e "\n${BLUE}=== Testing Health Endpoint ===${NC}"
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        print_status "PASS" "Health endpoint returns 200"
        
        # Check if response contains expected fields
        if echo "$body" | jq -e '.status' > /dev/null 2>&1; then
            print_status "PASS" "Health response contains status field"
        else
            print_status "FAIL" "Health response missing status field"
        fi
        
        if echo "$body" | jq -e '.embedding_service_initialized' > /dev/null 2>&1; then
            initialized=$(echo "$body" | jq -r '.embedding_service_initialized')
            if [ "$initialized" = "true" ]; then
                print_status "PASS" "Embedding service is initialized"
            else
                print_status "FAIL" "Embedding service is not initialized"
            fi
        else
            print_status "FAIL" "Health response missing embedding_service_initialized field"
        fi
    else
        print_status "FAIL" "Health endpoint returns $http_code"
    fi
}

# Test single text embedding
test_single_embedding() {
    echo -e "\n${BLUE}=== Testing Single Text Embedding ===${NC}"
    
    payload='{"text": "Hello world, this is a test sentence."}'
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/embeddings")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        print_status "PASS" "Single embedding endpoint returns 200"
        
        # Check response structure
        if echo "$body" | jq -e '.text' > /dev/null 2>&1; then
            print_status "PASS" "Response contains text field"
        else
            print_status "FAIL" "Response missing text field"
        fi
        
        if echo "$body" | jq -e '.embeddings' > /dev/null 2>&1; then
            embeddings_length=$(echo "$body" | jq '.embeddings | length')
            if [ "$embeddings_length" -gt 0 ]; then
                print_status "PASS" "Response contains embeddings array (length: $embeddings_length)"
            else
                print_status "FAIL" "Embeddings array is empty"
            fi
        else
            print_status "FAIL" "Response missing embeddings field"
        fi
        
        if echo "$body" | jq -e '.dimension' > /dev/null 2>&1; then
            dimension=$(echo "$body" | jq '.dimension')
            print_status "PASS" "Response contains dimension field (dimension: $dimension)"
        else
            print_status "FAIL" "Response missing dimension field"
        fi
    else
        print_status "FAIL" "Single embedding endpoint returns $http_code"
        echo "Response body: $body"
    fi
}

# Test batch text embedding
test_batch_embedding() {
    echo -e "\n${BLUE}=== Testing Batch Text Embedding ===${NC}"
    
    payload='{"texts": ["Hello world", "How are you today?", "This is a test sentence", "Machine learning is fascinating"]}'
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/embeddings")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        print_status "PASS" "Batch embedding endpoint returns 200"
        
        # Check response structure
        if echo "$body" | jq -e '.texts' > /dev/null 2>&1; then
            texts_length=$(echo "$body" | jq '.texts | length')
            print_status "PASS" "Response contains texts array (length: $texts_length)"
        else
            print_status "FAIL" "Response missing texts field"
        fi
        
        if echo "$body" | jq -e '.embeddings' > /dev/null 2>&1; then
            embeddings_count=$(echo "$body" | jq '.embeddings | length')
            if [ "$embeddings_count" -eq 4 ]; then
                print_status "PASS" "Response contains correct number of embeddings ($embeddings_count)"
            else
                print_status "FAIL" "Expected 4 embeddings, got $embeddings_count"
            fi
        else
            print_status "FAIL" "Response missing embeddings field"
        fi
        
        if echo "$body" | jq -e '.count' > /dev/null 2>&1; then
            count=$(echo "$body" | jq '.count')
            print_status "PASS" "Response contains count field (count: $count)"
        else
            print_status "FAIL" "Response missing count field"
        fi
    else
        print_status "FAIL" "Batch embedding endpoint returns $http_code"
        echo "Response body: $body"
    fi
}

# Test error handling
test_error_handling() {
    echo -e "\n${BLUE}=== Testing Error Handling ===${NC}"
    
    # Test invalid JSON
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d '{"invalid": json}' \
        "$BASE_URL/embeddings")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if echo "$body" | jq -e '.error' > /dev/null 2>&1; then
        error_msg=$(echo "$body" | jq -r '.error')
        print_status "PASS" "Invalid JSON returns error message: $error_msg"
    else
        print_status "FAIL" "Invalid JSON should return error message"
    fi
    
    # Test missing required fields
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d '{"invalid_field": "test"}' \
        "$BASE_URL/embeddings")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if echo "$body" | jq -e '.error' > /dev/null 2>&1; then
        error_msg=$(echo "$body" | jq -r '.error')
        print_status "PASS" "Missing required fields returns error message: $error_msg"
    else
        print_status "FAIL" "Missing required fields should return error message"
    fi
    
    # Test invalid data types
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": 123}' \
        "$BASE_URL/embeddings")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if echo "$body" | jq -e '.error' > /dev/null 2>&1; then
        error_msg=$(echo "$body" | jq -r '.error')
        print_status "PASS" "Invalid data type returns error message: $error_msg"
    else
        print_status "FAIL" "Invalid data type should return error message"
    fi
}

# Test performance with large text
test_performance() {
    echo -e "\n${BLUE}=== Testing Performance ===${NC}"
    
    # Create a longer text for performance testing
    long_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium."
    
    payload="{\"text\": \"$long_text\"}"
    
    start_time=$(date +%s.%N)
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/embeddings")
    end_time=$(date +%s.%N)
    
    http_code=$(echo "$response" | tail -n1)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if [ "$http_code" = "200" ]; then
        print_status "PASS" "Long text embedding completed in ${duration}s"
    else
        print_status "FAIL" "Long text embedding failed with HTTP $http_code"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Embedding API Test Suite${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Check dependencies
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl is required but not installed.${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq is required but not installed.${NC}"
        echo -e "${YELLOW}Install with: sudo apt-get install jq (Ubuntu/Debian) or brew install jq (macOS)${NC}"
        exit 1
    fi
    
    # Run tests
    check_server
    test_health
    test_single_embedding
    test_batch_embedding
    test_error_handling
    test_performance
    
    # Summary
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}           Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ Some tests failed.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
