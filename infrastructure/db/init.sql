-- Intelligent Document Classifier and Router Database Schema
-- PostgreSQL initialization script

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table - stores document metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_name VARCHAR(255) NOT NULL,
    storage_path TEXT NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    doc_type VARCHAR(50),
    confidence FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    tenant_id UUID DEFAULT uuid_generate_v4() -- For multi-tenancy
);

-- Metadata table - stores extracted information
CREATE TABLE metadata (
    id SERIAL PRIMARY KEY,
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    key_entities JSONB, -- Extracted names, dates, amounts
    key_phrases TEXT[],
    summary TEXT,
    related_docs UUID[],
    risk_score FLOAT DEFAULT 0.0,
    sentiment_score FLOAT,
    language VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Routing rules table - business logic for document routing
CREATE TABLE routing_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    condition JSONB NOT NULL, -- IF-THEN logic definition
    assignee VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    tenant_id UUID DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Routing decisions table - tracks routing history
CREATE TABLE routing_decisions (
    id SERIAL PRIMARY KEY,
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    rule_id INTEGER REFERENCES routing_rules(id),
    assigned_to VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 1,
    decision_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table - for authentication and assignment
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    expertise_tags TEXT[],
    is_active BOOLEAN DEFAULT true,
    tenant_id UUID DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Teams table - for team-based routing
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    members UUID[],
    tenant_id UUID DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table - for compliance and tracking
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processing queue table - for tracking document processing
CREATE TABLE processing_queue (
    id SERIAL PRIMARY KEY,
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'queued',
    priority INTEGER DEFAULT 1,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better performance
CREATE INDEX idx_documents_tenant_id ON documents(tenant_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_metadata_doc_id ON metadata(doc_id);
CREATE INDEX idx_routing_rules_tenant_id ON routing_rules(tenant_id);
CREATE INDEX idx_routing_decisions_doc_id ON routing_decisions(doc_id);
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_processing_queue_status ON processing_queue(status);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_routing_rules_updated_at BEFORE UPDATE ON routing_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO users (username, email, full_name, role, expertise_tags) VALUES
('admin', 'admin@example.com', 'System Administrator', 'admin', ARRAY['general']),
('legal_user', 'legal@example.com', 'Legal Team Member', 'legal', ARRAY['contracts', 'legal']),
('finance_user', 'finance@example.com', 'Finance Team Member', 'finance', ARRAY['invoices', 'financial']),
('hr_user', 'hr@example.com', 'HR Team Member', 'hr', ARRAY['hr', 'employment']);

-- Insert sample routing rules
INSERT INTO routing_rules (name, condition, assignee, priority) VALUES
('Legal Documents', '{"doc_type": "contract", "confidence": {"$gte": 0.7}}', 'legal_user', 1),
('Financial Documents', '{"doc_type": "invoice", "confidence": {"$gte": 0.7}}', 'finance_user', 1),
('HR Documents', '{"doc_type": "employment", "confidence": {"$gte": 0.7}}', 'hr_user', 1),
('Low Confidence Documents', '{"confidence": {"$lt": 0.7}}', 'admin', 5);

-- Create teams
INSERT INTO teams (name, description, members) VALUES
('Legal Team', 'Handles all legal documents and contracts', ARRAY[(SELECT id FROM users WHERE username = 'legal_user')]),
('Finance Team', 'Handles financial documents and invoices', ARRAY[(SELECT id FROM users WHERE username = 'finance_user')]),
('HR Team', 'Handles HR and employment documents', ARRAY[(SELECT id FROM users WHERE username = 'hr_user')]); 