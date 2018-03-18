from wtforms import Form, FloatField,StringField, validators
from math import pi

class InputForm(Form):
    x = StringField(label='Enter you tweet here',validators=[validators.InputRequired()])
    open = FloatField(
        label='Open', default=1.0,
        validators=[validators.InputRequired()])
    high = FloatField(
        label='High', default=0,
        validators=[validators.InputRequired()])
    low = FloatField(
        label='Low', default=2*pi,
        validators=[validators.InputRequired()])
    last = FloatField(
        label='Last', default=18,
        validators=[validators.InputRequired()])
	
	