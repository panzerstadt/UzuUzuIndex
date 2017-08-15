/**
 * Flipbook.js
 * ----------------------------------------------------------
 * Version: 1.0.2
 * Modified: 2012.10.16
 * Author: Kazy @ nbnote.jp
 * License: Dual licensed under the MIT and GPL licenses.
 * ----------------------------------------------------------
 */


(function( window, document ) {


	var Flipbook = function( screen, frames ) {

		this._screen = typeof screen === 'string' ? document.getElementById( screen ) :
		screen instanceof jQuery ? screen[0] :
		screen;

		this._frameList = [];
		this._style = this._screen.style;

		this._timer = 0;
		this._speed = 33;
		this._currentFrame = 0;
		this._prevFrame = 0;
		this._totalFrames = frames.length;
		this._loop = true;
		this._isPlaying = false;
		this._isReverse = false;

		this.onLastFrame = function( data ) {};
		this.onFirstFrame = function( data ) {};
		this.onUpdate = function( data ) {};

		if ( this._totalFrames ) {

			var fragment = document.createDocumentFragment(),
			frame, style,
			i = 0, len = this._totalFrames;

			for ( ; i < len; i++ ) {
				frame = new Image();
				frame.src = frames[i];
				style = frame.style;
				style.display = 'block';
				style.visibility = 'hidden';
				style.position = 'absolute';
				style.top = '0';
				style.left = '0';
				fragment.appendChild( frame );
				this._frameList[i] = frame;
			}

			this._screen.appendChild( fragment );
			this.setFrame( 0 );
		}
	};

	Flipbook.prototype = {

		play: function() {

			var that = this;

			if ( that._isPlaying && !that._isReverse || that._totalFrames < 2 ) {
				return;
			}

			that.pause();

			that._isPlaying = true;
			that._isReverse = false;

			that._timer = setInterval( function() {
				if ( !that._loop && that._currentFrame === that._totalFrames - 1 ) {
					that.pause();
				} else {
					that.setFrame( that._currentFrame + 1 );
				}
			}, that._speed );
			that.setFrame( that._currentFrame + 1 );
		},

		replay: function() {

			this.stop();
			if ( !this._isReverse ) {
				this.play();
			} else {
				this.reverse();
			}
		},

		reverse: function() {

			var that = this;

			if ( that._isPlaying && that._isReverse || that._totalFrames < 2 ) {
				return;
			}

			that.pause();

			that._isPlaying = true;
			that._isReverse = true;

			that._timer = setInterval( function() {
				if ( !that._loop && that._currentFrame === 0 ) {
					that.pause();
				} else {
					that.setFrame( that._currentFrame - 1 );
				}
			}, that._speed );
			that.setFrame( that._currentFrame - 1 );
		},

		pause: function() {

			if ( !this._isPlaying ) return;

			clearInterval( this._timer );
			this._isPlaying = false;
		},

		stop: function() {

			this.pause();
			this.setFrame( 0 );
		},

		setFrame: function( frameNumber ) {

			this._prevFrame = this._currentFrame;

			var current = this._currentFrame = frameNumber >= this._totalFrames ? 0 :
			frameNumber < 0 ? this._totalFrames - 1 :
			frameNumber;

			this._frameList[this._prevFrame].style.visibility = 'hidden';
			this._frameList[current].style.visibility = 'visible';

			this.onUpdate( { type: 'update', frameNumber: current } );
			if ( current === 0 ) {
				this.onFirstFrame( { type: 'first_frame', frameNumber: current } );
			} else if ( current === this._totalFrames - 1 ) {
				this.onLastFrame( { type: 'last_frame', frameNumber: current } );
			}
		},

		getSpeed: function() {

			return this._speed;
		},

		setSpeed: function( milliSecond ) {

			this._speed = milliSecond;
		},

		getLoop: function() {

			return this._loop;
		},

		setLoop: function( value ) {

			this._loop = value;
		},

		getCurrentFrame: function() {

			return this._currentFrame;
		},

		getTotalFrames: function() {

			return this._totalFrames;
		},

		isPlaying: function() {

			return this._isPlaying;
		},

		isReverse: function() {

			return this._isReverse;
		}

	};


	window.Flipbook = Flipbook;


}( window, document ));