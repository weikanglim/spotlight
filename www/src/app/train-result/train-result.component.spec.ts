import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainResultComponent } from './train-result.component';

describe('TrainResultComponent', () => {
  let component: TrainResultComponent;
  let fixture: ComponentFixture<TrainResultComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TrainResultComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TrainResultComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
