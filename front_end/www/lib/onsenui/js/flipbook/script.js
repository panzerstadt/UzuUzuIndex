$( function() {

	var flipbook,
	imgList = [
		'images/img_01.png',
		'images/img_02.png',
		'images/img_03.png',
		'images/img_04.png',
		'images/img_05.png',
		'images/img_06.png',
		'images/img_07.png',
		'images/img_08.png',
		'images/img_09.png',
		'images/img_10.png',
		'images/img_11.png',
		'images/img_12.png',
		'images/img_13.png',
		'images/img_14.png',
		'images/img_15.png',
		'images/img_16.png',
		'images/img_17.png',
		'images/img_18.png',
		'images/img_19.png',
		'images/img_20.png'
	],

	$frameNo = $( '#currentFrame' ),
	$knob = $( '#knob' ),
	btnList = [
		$( '#reverseBtn' ).data( { id: 0, methodName: 'reverse' } ),
		$( '#stopBtn' ).data( { id: 1, methodName: 'stop' } ),
		$( '#pauseBtn' ).data( { id: 2, methodName: 'pause' } ),
		$( '#playBtn' ).data( { id: 3, methodName: 'play' } )
	];


	var setState = function( id ) {
		for ( var i = btnList.length; i--; ) {
			var $btn = btnList[i];
			if ( $btn.data( 'id' ) === id ) {
				$btn[0].src = $btn[0].src.replace( '_off', '_on' );
			} else {
				$btn[0].src = $btn[0].src.replace( '_on', '_off' );
			}
		}
	};

	flipbook = new Flipbook( 'screen', imgList );
	flipbook.onLastFrame = flipbook.onFirstFrame = function() {
		if ( !flipbook.getLoop() ) {
			setState( 1 );
		}
	};
	flipbook.onUpdate = function( data ) {
		$frameNo.text( data.frameNumber );
	};

	for ( var i = btnList.length; i--; ) {
		btnList[i]
		.on( 'click', function() {
			var $btn = $( this );
			flipbook[$btn.data( 'methodName' )]();
			setState( $btn.data( 'id' ) );
		} );
	}

	$knob
	.on( 'click', function() {
		var isLoop = flipbook.getLoop();
		flipbook.setLoop( !isLoop );
		$knob
		.animate( {
			left: isLoop ? 10 : 23
		}, 200 );
	} );


} );


































