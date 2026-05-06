"""
Medical Prescription System - Flask Web Application
Features: User authentication, prescription history, medicine detection, PDF export
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import os
import json
from datetime import datetime
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from werkzeug.security import generate_password_hash
import time
import sys

from ocr_pipeline import process_image
import importlib

# Force reload OCR pipeline to get latest changes
importlib.reload(sys.modules['ocr_pipeline'])

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'medscript-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///med_script.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# Create uploads directory
os.makedirs('uploads', exist_ok=True)

# Import and initialize database
from models import db, User, PrescriptionHistory, DetectedMedicine

db.init_app(app)



# ============== AUTHENTICATION HELPERS ==============

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get currently logged in user"""
    if 'user_id' in session:
        return db.session.get(User, session['user_id'])
    return None


# ============== ROUTES: AUTHENTICATION ==============

@app.route('/')
def index():
    """Home page - redirect based on login status"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            return render_template('register.html', error='All fields required')
        
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')
        
        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            return redirect(url_for('login', success='Registration successful! Please login.'))
        except Exception as e:
            db.session.rollback()
            return render_template('register.html', error=f'Registration failed: {str(e)}')
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            return render_template('login.html', error='Username and password required')
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        
        return render_template('login.html', error='Invalid username or password')
    
    success = request.args.get('success', '')
    return render_template('login.html', success=success)


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('login'))


# ============== ROUTES: MAIN APPLICATION ==============

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with history"""
    user = get_current_user()
    
    # Get user's prescription history
    prescriptions = PrescriptionHistory.query.filter_by(user_id=user.id).order_by(
        PrescriptionHistory.upload_date.desc()
    ).all()
    
    # Statistics
    stats = {
        'total_uploads': len(prescriptions),
        'total_medicines': sum(p.total_medicines for p in prescriptions),
        'avg_medicines': sum(p.total_medicines for p in prescriptions) / len(prescriptions) if prescriptions else 0
    }
    
    return render_template('dashboard.html', user=user, prescriptions=prescriptions, stats=stats)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload and process prescription image"""
    user = get_current_user()
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            return render_template('upload.html', error='Only image files allowed (JPG, PNG, etc.)')
        
        try:
            # Save uploaded file
            upload_dir = os.path.join('uploads', str(user.id))
            os.makedirs(upload_dir, exist_ok=True)
            
            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            
            # Process image with medicine detection
            start_time = time.time()
            detected_medicines = process_image(filepath)
            processing_time = time.time() - start_time
            
            # No fallback system - only OCR results
            fallback_was_used = False
            prescription_id_detected = ""
            
            # Create prescription history record
            prescription = PrescriptionHistory(
                user_id=user.id,
                filename=filename,
                total_medicines=len(detected_medicines),
                processing_time=processing_time,
                fallback_used=False,
                prescription_id_detected=""
            )
            
            # Store raw medicines as JSON
            medicines_json = json.dumps(detected_medicines)
            prescription.detected_medicines = medicines_json
            
            db.session.add(prescription)
            db.session.flush()  # Get the prescription ID
            
            # Store individual medicines
            for med in detected_medicines:
                detected_med = DetectedMedicine(
                    prescription_id=prescription.id,
                    medicine_name=med.get('medicine', ''),
                    dosage=med.get('dosage', ''),
                    frequency=med.get('frequency', ''),
                    duration=med.get('duration', ''),
                    confidence=med.get('confidence', 0.0),
                    note=med.get('note', '')
                )
                db.session.add(detected_med)
            
            db.session.commit()
            
            # Redirect to results page
            return redirect(url_for('results', prescription_id=prescription.id))
        
        except Exception as e:
            db.session.rollback()
            return render_template('upload.html', error=f'Processing failed: {str(e)}')
    
    return render_template('upload.html')


@app.route('/results/<int:prescription_id>')
@login_required
def results(prescription_id):
    """Display detection results"""
    user = get_current_user()
    prescription = db.session.get(PrescriptionHistory, prescription_id)
    
    # Security: ensure user owns this prescription
    if not prescription or prescription.user_id != user.id:
        return render_template('error.html', error='Prescription not found or access denied'), 404
    
    # Get detected medicines
    medicines = DetectedMedicine.query.filter_by(prescription_id=prescription_id).all()
    
    # Sort by confidence descending
    medicines.sort(key=lambda x: x.confidence, reverse=True)
    
    return render_template('results.html', prescription=prescription, medicines=medicines)


@app.route('/prescription-image/<int:prescription_id>')
@login_required
def get_prescription_image(prescription_id):
    """Get prescription image for display"""
    user = get_current_user()
    prescription = db.session.get(PrescriptionHistory, prescription_id)
    
    # Security check
    if not prescription or prescription.user_id != user.id:
        return "Access denied", 403
    
    # Build file path
    filepath = os.path.join('uploads', str(user.id), prescription.filename)
    
    if not os.path.exists(filepath):
        return "Image not found", 404
    
    return send_file(filepath, mimetype='image/jpeg')


@app.route('/history')
@login_required
def history():
    """View complete prescription history"""
    user = get_current_user()
    
    page = request.args.get('page', 1, type=int)
    prescriptions = PrescriptionHistory.query.filter_by(user_id=user.id).order_by(
        PrescriptionHistory.upload_date.desc()
    ).paginate(page=page, per_page=10)
    
    return render_template('history.html', prescriptions=prescriptions)


# ============== ROUTES: EXPORT ==============

@app.route('/download-pdf/<int:prescription_id>')
@login_required
def download_pdf(prescription_id):
    """Download prescription as PDF"""
    user = get_current_user()
    prescription = db.session.get(PrescriptionHistory, prescription_id)
    
    # Security check
    if not prescription or prescription.user_id != user.id:
        return "Access denied", 403
    
    # Get medicines
    medicines = DetectedMedicine.query.filter_by(prescription_id=prescription_id).all()
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    
    try:
        # Create PDF
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#007bff'),
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("💊 Digital Prescription", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient info
        story.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
        story.append(Paragraph(f"<b>Username:</b> {user.username}", styles['Normal']))
        story.append(Paragraph(f"<b>Email:</b> {user.email}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Medicines table
        if medicines:
            story.append(Paragraph("<b>Prescribed Medicines</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            # Build table data
            table_data = [['Medicine', 'Dosage', 'Frequency', 'Duration', 'Confidence']]
            for med in medicines:
                table_data.append([
                    med.medicine_name,
                    med.dosage or '—',
                    med.frequency or '—',
                    med.duration or '—',
                    f"{med.confidence:.0%}"
                ])
            
            # Create table
            table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1*inch, 1.1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007bff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.3*inch))
            
            # Add confidence legend
            story.append(Paragraph("<font size=9><b>Confidence Levels:</b><br/>🟢 HIGH (80%+) - Medicine clearly detected<br/>🟡 MEDIUM (60-79%) - Partial detection<br/>🔴 LOW (20-59%) - Requires verification</font>", styles['Normal']))
        else:
            story.append(Paragraph("❌ No medicines detected in this prescription", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        story.append(Paragraph(
            "<font size=8 color='red'><b>⚠️ DISCLAIMER:</b> This is an automated medicine analysis document. "
            "Always verify with a qualified healthcare professional before taking any medication. "
            "This system is not a substitute for professional medical advice.</font>",
            styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        
        pdf_buffer.seek(0)
        
        # Send file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'prescription_{prescription_id}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500


@app.route('/download-csv/<int:prescription_id>')
@login_required
def download_csv(prescription_id):
    """Download prescription as CSV"""
    user = get_current_user()
    prescription = db.session.get(PrescriptionHistory, prescription_id)
    
    # Security check
    if not prescription or prescription.user_id != user.id:
        return "Access denied", 403
    
    # Get medicines
    medicines = DetectedMedicine.query.filter_by(prescription_id=prescription_id).all()
    
    # Build CSV
    csv_content = "Medicine,Dosage,Frequency,Duration,Confidence\n"
    for med in medicines:
        csv_content += f"{med.medicine_name},{med.dosage or ''},"\
                       f"{med.frequency or ''},{med.duration or ''}," \
                       f"{med.confidence:.2%}\n"
    
    return send_file(
        io.BytesIO(csv_content.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'prescription_{prescription_id}.csv'
    )


@app.route('/view-prescription/<int:prescription_id>')
@login_required
def view_prescription(prescription_id):
    """API endpoint to view prescription details (for AJAX)"""
    user = get_current_user()
    prescription = db.session.get(PrescriptionHistory, prescription_id)
    
    if not prescription or prescription.user_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    medicines = DetectedMedicine.query.filter_by(prescription_id=prescription_id).all()
    
    medicines_data = []
    for med in medicines:
        medicines_data.append({
            'name': med.medicine_name,
            'dosage': med.dosage,
            'frequency': med.frequency,
            'duration': med.duration,
            'confidence': f"{med.confidence:.0%}"
        })
    
    return jsonify({
        'prescription_id': prescription.id,
        'filename': prescription.filename,
        'upload_date': prescription.upload_date.strftime('%Y-%m-%d %H:%M'),
        'medicines': medicines_data,
        'total_medicines': len(medicines)
    })


# ============== ERROR HANDLERS ==============

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error='Internal server error'), 500


# ============== INITIALIZATION ==============

@app.before_request
def create_tables():
    """Create database tables on first run"""
    if not hasattr(app, 'tables_created'):
        db.create_all()
        app.tables_created = True


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # use_reloader=False prevents signal handler issues on Windows
    app.run(debug=True, port=5000, use_reloader=False)
