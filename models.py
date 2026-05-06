"""
Database Models for Medical Prescription System
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """User account model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to prescriptions
    prescriptions = db.relationship('PrescriptionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class PrescriptionHistory(db.Model):
    """Prescription upload and detection history"""
    __tablename__ = 'prescription_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # File information
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Detection results (stored as JSON string)
    raw_ocr_text = db.Column(db.Text)  # Raw text extracted from image
    detected_medicines = db.Column(db.Text)  # JSON string of detected medicines
    total_medicines = db.Column(db.Integer, default=0)
    
    # Metadata
    ocr_quality = db.Column(db.Float, default=0.0)  # Quality score of image extraction
    processing_time = db.Column(db.Float, default=0.0)  # Time taken to process in seconds
    fallback_used = db.Column(db.Boolean, default=False)  # Whether fallback dataset was used
    prescription_id_detected = db.Column(db.String(10), default="")  # Detected prescription ID (P1, P2, etc.)
    
    def __repr__(self):
        return f'<PrescriptionHistory {self.id} - {self.filename}>'


class DetectedMedicine(db.Model):
    """Detected medicine from a prescription"""
    __tablename__ = 'detected_medicines'
    
    id = db.Column(db.Integer, primary_key=True)
    prescription_id = db.Column(db.Integer, db.ForeignKey('prescription_history.id'), nullable=False)
    
    # Medicine details
    medicine_name = db.Column(db.String(255), nullable=False)
    dosage = db.Column(db.String(255), default="")
    frequency = db.Column(db.String(255), default="")
    duration = db.Column(db.String(255), default="")
    
    # Confidence and notes
    confidence = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    note = db.Column(db.Text, default="")
    
    def __repr__(self):
        return f'<DetectedMedicine {self.medicine_name} - {self.confidence:.2f}>'
